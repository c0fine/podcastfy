"""Local web UI: drop in a paper, get audio, browse the library.

Runs against Ollama (transcript) and Kokoro (audio), both reached over HTTP, so
it works the same on the host or inside Docker (see docker-compose.yml).
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from podcastfy.client import process_content
from podcastfy.read_aloud import read_aloud
from podcastfy.utils.config_conversation import load_conversation_config

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("PODCASTFY_DATA_DIR", "data"))
AUDIO_DIR = DATA_DIR / "audio"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = Path(__file__).parent / "static"
TEXT_SUFFIXES = {".txt", ".md"}
CHUNK = 1024 * 256

app = FastAPI(title="Podcastfy local")
# Ollama and Kokoro handle one request at a time, so jobs run one after another.
executor = ThreadPoolExecutor(max_workers=1)
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


def _update(job_id: str, **fields: Any) -> None:
    with jobs_lock:
        jobs[job_id].update(fields)


def _duration(path: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def _safe_name(name: str) -> Path:
    """Resolve an audio file name inside AUDIO_DIR, rejecting anything else."""
    if not re.fullmatch(r"[\w.\- ]+\.mp3", name) or ".." in name:
        raise HTTPException(status_code=404, detail="Not found")
    path = AUDIO_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return path


def _run_job(job_id: str, source: Dict[str, Any], opts: Dict[str, Any]) -> None:
    _update(job_id, status="running", started=time.time())
    try:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        urls = [source["path"]] if source.get("path") else None
        text = source.get("text")

        if opts["mode"] == "read":
            kokoro = load_conversation_config().get("text_to_speech", {}).get("kokoro", {})
            audio_file = read_aloud(
                urls=urls,
                text=text,
                voice=opts["voice"],
                model=kokoro.get("model"),
                output_dir=str(AUDIO_DIR),
                skip_references=opts["skip_references"],
                progress=lambda done, total: _update(
                    job_id, progress=done / total, detail=f"{done}/{total} chunks"
                ),
            )
        else:
            _update(job_id, detail="Writing the script with Ollama, then voicing it")
            audio_file = process_content(
                urls=urls,
                text=text,
                tts_model="kokoro",
                is_local=True,
                model_name=opts["llm_model"] or None,
                longform=opts["longform"],
            )

        audio_path = Path(audio_file)
        if audio_path.parent.resolve() != AUDIO_DIR.resolve():
            target = AUDIO_DIR / audio_path.name
            audio_path.replace(target)
            audio_path = target
        meta = {
            "title": source["title"],
            "mode": opts["mode"],
            "created": time.time(),
            "duration": _duration(audio_path),
        }
        audio_path.with_suffix(".json").write_text(json.dumps(meta))
        _update(job_id, status="done", progress=1.0, file=audio_path.name, detail="")
    except Exception as e:  # surfaced to the UI
        logger.exception("Job failed")
        _update(job_id, status="error", error=str(e))


@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    url: str = Form(""),
    text: str = Form(""),
    mode: str = Form("read"),
    voice: str = Form("af_heart"),
    skip_references: bool = Form(False),
    longform: bool = Form(False),
    llm_model: str = Form(""),
):
    if mode not in ("read", "podcast"):
        raise HTTPException(status_code=400, detail="mode must be 'read' or 'podcast'")

    source: Dict[str, Any]
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf"} | TEXT_SUFFIXES:
            raise HTTPException(status_code=400, detail="Upload a PDF, .txt or .md file")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        saved.write_bytes(await file.read())
        title = Path(file.filename).stem
        if suffix == ".pdf":
            source = {"path": str(saved), "title": title}
        else:
            source = {"text": saved.read_text(errors="replace"), "title": title}
    elif url.strip():
        source = {"path": url.strip(), "title": url.strip()}
    elif text.strip():
        source = {"text": text, "title": text.strip()[:60]}
    else:
        raise HTTPException(status_code=400, detail="Provide a file, URL or text")

    opts = {
        "mode": mode,
        "voice": voice,
        "skip_references": skip_references,
        "longform": longform,
        "llm_model": llm_model.strip(),
    }
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id, "title": source["title"], "mode": mode,
            "status": "queued", "progress": None, "detail": "", "created": time.time(),
        }
    executor.submit(_run_job, job_id, source, opts)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        return sorted(jobs.values(), key=lambda j: j["created"], reverse=True)


@app.get("/api/library")
def library():
    items = []
    for mp3 in AUDIO_DIR.glob("*.mp3") if AUDIO_DIR.is_dir() else []:
        meta: Dict[str, Any] = {}
        sidecar = mp3.with_suffix(".json")
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text())
            except ValueError:
                pass
        stat = mp3.stat()
        items.append({
            "file": mp3.name,
            "title": meta.get("title") or mp3.stem,
            "mode": meta.get("mode"),
            "duration": meta.get("duration"),
            "created": meta.get("created") or stat.st_mtime,
            "size": stat.st_size,
        })
    return sorted(items, key=lambda i: i["created"], reverse=True)


@app.delete("/api/library/{name}")
def delete_audio(name: str):
    path = _safe_name(name)
    path.unlink()
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"deleted": name}


@app.get("/audio/{name}")
def serve_audio(name: str, request: Request):
    """Serve an MP3 with Range support so long recordings can be seeked."""
    path = _safe_name(name)
    size = path.stat().st_size
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", request.headers.get("range", ""))
    if not match or not (match.group(1) or match.group(2)):
        return FileResponse(path, media_type="audio/mpeg",
                            headers={"Accept-Ranges": "bytes"})
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:  # suffix range: last N bytes
        start, end = max(size - int(match.group(2)), 0), size - 1
    end = min(end, size - 1)
    if start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def stream():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        stream(), status_code=206, media_type="audio/mpeg",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
            "Accept-Ranges": "bytes",
        },
    )


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()
