"""Read a document aloud, verbatim, with a single voice (no LLM involved)."""

import io
import logging
import os
import re
import uuid
from typing import List, Optional

from pydub import AudioSegment

from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.tts.providers.kokoro import KokoroTTS

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 800
PARAGRAPH_PAUSE_MS = 500


def clean_text(text: str) -> str:
    """Undo PDF line wrapping: re-join hyphenated words and soft-wrapped lines."""
    text = text.replace("\r", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def strip_references(text: str) -> str:
    """Cut everything from the last standalone 'References'/'Bibliography' heading on.

    Ignored if the heading sits in the first third of the text (likely a table of
    contents entry), so short documents are never emptied.
    """
    matches = list(
        re.finditer(r"^\s*(references|bibliography)\s*$", text, re.I | re.M)
    )
    if matches and matches[-1].start() > len(text) / 3:
        return text[: matches[-1].start()]
    return text


def split_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks of whole sentences, each at most max_chars long.

    An empty string in the result marks a paragraph boundary.
    """
    chunks: List[str] = []
    for paragraph in clean_text(text).split("\n\n"):
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            while len(sentence) > max_chars:  # no sentence break: cut at a space
                cut = sentence.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if current and len(current) + 1 + len(sentence) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
        chunks.append("")
    return chunks[:-1]


def read_aloud(
    urls: Optional[List[str]] = None,
    text: Optional[str] = None,
    output_file: Optional[str] = None,
    voice: str = "af_heart",
    model: Optional[str] = None,
    output_dir: str = "data/audio",
    skip_references: bool = False,
) -> str:
    """Extract text from the sources and read it, as written, with one Kokoro voice."""
    parts: List[str] = []
    if urls:
        extractor = ContentExtractor()
        parts.extend(extractor.extract_content(u) for u in urls)
    if text:
        parts.append(text)
    body = "\n\n".join(parts)
    if skip_references:
        body = strip_references(body)
    chunks = split_chunks(body)
    if not any(chunks):
        raise ValueError("No text found to read aloud.")

    tts = KokoroTTS(model=model)
    combined = AudioSegment.empty()
    total = sum(1 for c in chunks if c)
    done = 0
    for chunk in chunks:
        if not chunk:
            combined += AudioSegment.silent(duration=PARAGRAPH_PAUSE_MS)
            continue
        audio = tts.generate_audio(chunk, voice, tts.model)
        combined += AudioSegment.from_file(io.BytesIO(audio), format="mp3")
        done += 1
        logger.info(f"Read {done}/{total} chunks")

    if output_file is None:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"reading_{uuid.uuid4().hex}.mp3")
    combined.export(output_file, format="mp3")
    return output_file
