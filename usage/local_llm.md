# Local LLM and TTS Support

Podcastfy can run with open-source models on your own machine: [Ollama](https://ollama.com) writes the transcript and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) produces the audio. No API keys are needed.

Running locally offers:
- Enhanced privacy and data security
- Cost control and no API rate limits
- Reduced vendor lock-in

## Setup

### 1. Ollama (transcript)

```bash
brew install ollama        # or download from https://ollama.com
ollama serve               # listens at http://localhost:11434
ollama pull llama3.1       # any chat model you prefer
```

### 2. Kokoro (audio)

Kokoro is reached through any OpenAI-compatible speech server. On Apple Silicon, [mlx-audio](https://github.com/Blaizzy/mlx-audio) works well:

```bash
brew install ffmpeg espeak-ng   # without espeak-ng the server crashes on words missing from its dictionary (e.g. names)
pip install "mlx-audio[server]" misaki num2words spacy phonemizer-fork espeakng-loader
mlx_audio.server --host 127.0.0.1 --port 8000
```

## Python API

```python
from podcastfy.client import generate_podcast

generate_podcast(
    urls=["https://example.com/article"],
    llm_model_name="ollama/llama3.1",  # or is_local=True to use the configured model
    tts_model="kokoro",
)
```

## CLI

```bash
python -m podcastfy.client --url https://example.com/article \
  --local --llm-model-name llama3.1 --tts-model kokoro
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| Ollama model | `content_generator.ollama.model` in `config.yaml` | `llama3.1` |
| Ollama address | `content_generator.ollama.api_base` in `config.yaml`, or `OLLAMA_API_BASE` | `http://localhost:11434` |
| Context window (tokens) | `content_generator.ollama.num_ctx` in `config.yaml` | `16384` |
| Longform context cap (characters) | `content_generator.ollama.max_context_chars` in `config.yaml` | `12000` |
| Kokoro voices and model | `text_to_speech.kokoro` in the conversation config | `af_heart`, `am_michael`, `mlx-community/Kokoro-82M-bf16` |
| Kokoro server address | `KOKORO_BASE_URL` | `http://localhost:8000/v1` |

To make Kokoro the default, set `text_to_speech.default_tts_model: "kokoro"` in your conversation config.

## Limitations

1. Topic-based generation (`topic=` / `--topic`) relies on Gemini with Google Search and is not available with a local LLM.
2. Images are ignored: a local LLM is assumed to be text-only.
3. Prompt templates are still downloaded from LangChain Hub, so an internet connection is needed when a run starts.
4. Smaller models sometimes ignore the `<Person1>`/`<Person2>` transcript format. Podcastfy raises an error when a transcript has no usable turns; rerun or use a larger model. For important work, generate with `--transcript-only` first and check the transcript before producing audio.
5. Keep `audio_format: "mp3"` when using Kokoro.

## Web UI (drag in a paper, browse generated audio)

```bash
ollama serve &                                   # on the host
mlx_audio.server --host 127.0.0.1 --port 8000 &  # on the host
docker compose up -d --build web
```

Open http://localhost:8080, drop in a PDF, and choose **Read aloud** (the paper as written, one voice,
optionally stopping at References) or **Two-host podcast** (Ollama writes a script). Finished audio is
listed in the library below the form and is stored in `./data/audio`.

Ollama and Kokoro stay on the host because MLX needs Apple Silicon and Linux containers cannot use the GPU.
The container reaches them through `host.docker.internal`; override with `OLLAMA_API_BASE` and `KOKORO_BASE_URL`.
Without Docker: `uvicorn podcastfy.web.app:app --port 8080`.
