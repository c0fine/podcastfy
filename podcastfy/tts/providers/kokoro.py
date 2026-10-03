"""Kokoro TTS provider implementation.

Talks to a local OpenAI-compatible speech server (for example mlx-audio on
Apple Silicon, or Kokoro-FastAPI). No API key is needed.
"""

import os
import subprocess
from typing import List, Optional

import requests

from ..base import TTSProvider

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"


class KokoroTTS(TTSProvider):
    """Kokoro Text-to-Speech provider served by a local OpenAI-compatible server."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Kokoro TTS provider.

        Args:
            api_key: Not needed for a local server. Sent as a bearer token if given.
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("KOKORO_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.base_url = os.environ.get("KOKORO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(os.environ.get("KOKORO_TIMEOUT", "300"))

    def get_supported_tags(self) -> List[str]:
        """Kokoro reads plain text, so no SSML tags are kept."""
        return []

    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        """Generate MP3 audio using the local Kokoro server."""
        model = model or DEFAULT_MODEL
        self.validate_parameters(text, voice, model)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.post(
                f"{self.base_url}/audio/speech",
                json={
                    "model": model,
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                },
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Could not reach the Kokoro server at {self.base_url}. "
                "Start it (e.g. `mlx_audio.server --port 8000`) or set KOKORO_BASE_URL."
            ) from e
        except requests.RequestException as e:
            detail = e.response.text[:300] if e.response is not None else str(e)
            raise RuntimeError(f"Failed to generate audio: {detail}") from e

        audio = response.content
        if not audio:
            raise RuntimeError("Kokoro server returned no audio")
        return self._ensure_mp3(audio)

    @staticmethod
    def _ensure_mp3(audio: bytes) -> bytes:
        """Convert WAV output to MP3; some servers ignore the requested format."""
        if not audio.startswith(b"RIFF"):
            return audio
        try:
            result = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
                input=audio,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg is required to convert Kokoro audio to MP3") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg failed to convert Kokoro audio: {e.stderr.decode(errors='ignore')[:300]}"
            ) from e
        return result.stdout
