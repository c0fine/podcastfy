"""Tests for the local stack: Ollama for the transcript and Kokoro for the audio.

No model or server is needed; Ollama is mocked and Kokoro is replaced by a tiny
HTTP server that answers like an OpenAI-compatible speech endpoint.
"""

import io
import json
import shutil
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from podcastfy.client import process_content
from podcastfy.content_generator import (
    ContentGenerator,
    LLMBackend,
    LongFormContentGenerator,
)
from podcastfy.text_to_speech import TextToSpeech
from podcastfy.tts.providers.kokoro import KokoroTTS


def _wav_bytes(seconds: float = 0.2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * int(24000 * seconds))
    return buffer.getvalue()


class _SpeechHandler(BaseHTTPRequestHandler):
    requests_seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _SpeechHandler.requests_seen.append((self.path, body))
        payload = _wav_bytes()  # like a server that ignores response_format
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestOllamaBackend(unittest.TestCase):
    @patch("langchain_community.chat_models.ChatOllama")
    def test_local_backend_uses_ollama(self, chat_ollama):
        LLMBackend(
            is_local=True,
            temperature=0.7,
            max_output_tokens=2048,
            model_name="ollama/llama3.1",
            api_key_label=None,
            num_ctx=16384,
        )
        kwargs = chat_ollama.call_args.kwargs
        self.assertEqual(kwargs["model"], "llama3.1")
        self.assertEqual(kwargs["base_url"], "http://localhost:11434")
        self.assertEqual(kwargs["num_predict"], 2048)
        self.assertEqual(kwargs["num_ctx"], 16384)

    @patch("podcastfy.content_generator.ChatLiteLLM")
    def test_remote_backend_needs_no_key(self, chat_litellm):
        LLMBackend(
            is_local=False,
            temperature=0.7,
            max_output_tokens=2048,
            model_name="openai/some-model",
            api_key_label=None,
        )
        self.assertIsNone(chat_litellm.call_args.kwargs["api_key"])

    @patch("langchain_community.chat_models.ChatOllama")
    def test_model_prefix_selects_ollama(self, chat_ollama):
        generator = ContentGenerator(model_name="ollama/qwen3", api_key_label=None)
        self.assertTrue(generator.is_local)
        self.assertEqual(chat_ollama.call_args.kwargs["model"], "qwen3")

    @patch("langchain_community.chat_models.ChatOllama")
    def test_local_flag_uses_configured_model(self, chat_ollama):
        generator = ContentGenerator(is_local=True, api_key_label=None)
        configured = generator.content_generator_config["ollama"]["model"]
        self.assertEqual(chat_ollama.call_args.kwargs["model"], configured)

    def test_longform_context_is_capped(self):
        generator = LongFormContentGenerator(None, None, {}, max_context_chars=60)
        context = "<Person1>" + "a" * 100 + "</Person1><Person2>short reply</Person2>"
        capped = generator._cap_context(context)
        self.assertLessEqual(len(capped), 60)
        self.assertTrue(capped.startswith("<Person2>"))
        self.assertEqual(len(generator._cap_context("x" * 100, keep_tail=False)), 60)
        unlimited = LongFormContentGenerator(None, None, {})
        self.assertEqual(unlimited._cap_context(context), context)

    def test_topic_rejected_when_local(self):
        with self.assertRaises(ValueError):
            process_content(topic="anything", is_local=True, generate_audio=False)


class TestKokoroProvider(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _SpeechHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _SpeechHandler.requests_seen.clear()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_generates_mp3_from_wav_response(self):
        with patch.dict("os.environ", {"KOKORO_BASE_URL": self.base_url}):
            provider = KokoroTTS()
        audio = provider.generate_audio("Hello there", "af_heart", "kokoro-model")
        self.assertFalse(audio.startswith(b"RIFF"))
        self.assertTrue(audio[:3] == b"ID3" or audio[0] == 0xFF)
        path, body = _SpeechHandler.requests_seen[0]
        self.assertEqual(path, "/v1/audio/speech")
        self.assertEqual(body["voice"], "af_heart")
        self.assertEqual(body["model"], "kokoro-model")
        self.assertEqual(body["input"], "Hello there")

    def test_unreachable_server_gives_clear_error(self):
        with patch.dict("os.environ", {"KOKORO_BASE_URL": "http://127.0.0.1:9/v1"}):
            provider = KokoroTTS()
        with self.assertRaises(RuntimeError) as raised:
            provider.generate_audio("Hello", "af_heart", "kokoro-model")
        self.assertIn("Could not reach the Kokoro server", str(raised.exception))

    def test_malformed_transcript_is_rejected(self):
        with patch.dict("os.environ", {"KOKORO_BASE_URL": self.base_url}):
            tts = TextToSpeech(model="kokoro")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                tts._generate_audio_segments("Host: hi\nGuest: hello", temp_dir)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_transcript_to_podcast_file(self):
        transcript = (
            "<Person1>Welcome to the show.</Person1><Person2>Glad to be here.</Person2>"
            "<Person1>Let's begin.</Person1><Person2>Sure.</Person2>"
        )
        with patch.dict("os.environ", {"KOKORO_BASE_URL": self.base_url}):
            tts = TextToSpeech(model="kokoro")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = f"{temp_dir}/podcast.mp3"
            tts.convert_to_speech(transcript, output)
            with open(output, "rb") as audio_file:
                self.assertGreater(len(audio_file.read()), 0)
        voices = [body["voice"] for _, body in _SpeechHandler.requests_seen]
        self.assertEqual(voices, ["af_heart", "am_michael"] * 2)
        self.assertEqual(
            _SpeechHandler.requests_seen[0][1]["model"], "mlx-community/Kokoro-82M-bf16"
        )


if __name__ == "__main__":
    unittest.main()


def test_read_aloud_chunks_keep_text_verbatim():
    from podcastfy.read_aloud import split_chunks

    text = "A well-\nknown result. It holds.\n\nSecond paragraph here."
    chunks = split_chunks(text, max_chars=30)
    assert "" in chunks  # paragraph boundary
    assert " ".join(c for c in chunks if c).replace("  ", " ") == (
        "A wellknown result. It holds. Second paragraph here."
    )
    assert all(len(c) <= 30 for c in chunks)


def test_strip_references_cuts_at_last_heading_only_when_late():
    from podcastfy.read_aloud import strip_references

    body = "Intro text. " * 50
    assert strip_references(f"{body}\nReferences\n[1] Foo") == f"{body}\n"
    assert strip_references("References\nShort doc body " * 1) == "References\nShort doc body "
