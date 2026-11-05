"""Module 1 - multimodal ingestion & normalisation.

Turns .pdf / .docx / .mp4 into a list of `Segment`s (text + page or timestamp
metadata).  Every parser is wrapped so that a corrupt, empty, or silent file
yields an `ExtractedDocument` with `ok=False` and human-readable warnings
instead of an exception.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import unicodedata
import wave
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger("integrity.extractor")

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".mp4": "video"}
DOCX_WORDS_PER_PAGE = 450  # DOCX has no fixed pagination: estimate (flagged)


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    text: str
    page: Optional[int] = None          # PDF page (exact) / DOCX page (estimated)
    start: Optional[float] = None       # seconds (video)
    end: Optional[float] = None


@dataclass
class ExtractedDocument:
    doc_id: str
    path: str
    modality: str                       # pdf | docx | video | unknown
    segments: List[Segment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    def fail(self, msg: str) -> "ExtractedDocument":
        self.ok = False
        self.error = msg
        self.segments = []
        self.warnings.append(msg)
        log.warning("%s: %s", self.doc_id, msg)
        return self


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "’": "'",
              "‘": "'", "“": '"', "”": '"', "–": "-",
              "—": "-", " ": " ", "​": ""}


def normalize_text(text: str) -> str:
    """Unicode-normalise, repair hyphenated line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # re-join "inte-\ngration"
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fmt_ts(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def fmt_range(start: float, end: float) -> str:
    return f"{fmt_ts(start)} - {fmt_ts(end)}"


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
_PAGE_NUM = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)


def _strip_repeated_lines(pages: List[List[str]]) -> List[List[str]]:
    """Remove running headers/footers: lines repeated on >=50% of pages
    (digits masked so 'Page 1' == 'Page 2') plus bare page numbers."""
    n = len(pages)

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter = Counter()
    for lines in pages:
        counts.update({key(ln) for ln in lines if ln.strip()})
    out = []
    for lines in pages:
        kept = []
        for ln in lines:
            s = ln.strip()
            if not s or _PAGE_NUM.match(s):
                continue
            if n >= 3 and counts[key(ln)] >= max(2, 0.5 * n):
                continue
            kept.append(ln)
        out.append(kept)
    return out


def extract_pdf(path: str, doc: ExtractedDocument,
                margin: float = 0.05) -> ExtractedDocument:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            return doc.fail("PDF has no pages")
        page_lines: List[List[str]] = []
        for page in pdf.pages:
            try:
                h, w = float(page.height), float(page.width)
                body = page.crop((0, h * margin, w, h * (1 - margin)))  # drop top/bottom 5%
                txt = body.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - per-page isolation
                doc.warnings.append(f"page {page.page_number}: {exc}")
                txt = ""
            page_lines.append(txt.split("\n"))
    page_lines = _strip_repeated_lines(page_lines)
    for i, lines in enumerate(page_lines, start=1):
        text = normalize_text("\n".join(lines))
        if text:
            doc.segments.append(Segment(text=text, page=i))
    if not doc.segments:
        return doc.fail("PDF contains no extractable text (scanned image or empty)")
    return doc


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def extract_docx(path: str, doc: ExtractedDocument) -> ExtractedDocument:
    import docx

    d = docx.Document(path)
    blocks: List[str] = []
    # keep body order: paragraphs and tables interleaved
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            t = Paragraph(child, d).text.strip()
            if t:
                blocks.append(t)
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells = []
                for c in row.cells:
                    if c.text.strip() and c.text.strip() not in cells:
                        cells.append(c.text.strip())
                if cells:
                    blocks.append(" | ".join(cells))
    if not blocks:
        return doc.fail("DOCX contains no text")
    # DOCX stores no pagination -> estimate pages from word count (flagged)
    page, words = 1, 0
    for b in blocks:
        doc.segments.append(Segment(text=normalize_text(b), page=page))
        words += len(b.split())
        if words >= DOCX_WORDS_PER_PAGE:
            page, words = page + 1, 0
    doc.meta["page_numbers_estimated"] = True
    return doc


# --------------------------------------------------------------------------- #
# MP4 -> audio -> transcript
# --------------------------------------------------------------------------- #
def _run(cmd: List[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _has_audio_stream(path: str) -> bool:
    r = _run(["ffmpeg", "-hide_banner", "-i", path])  # prints stream info to stderr
    return bool(re.search(r"Stream #\d+:\d+.*Audio:", r.stderr))


def extract_audio(path: str, wav_path: str) -> None:
    r = _run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
              "-ar", "16000", "-c:a", "pcm_s16le", wav_path])
    if r.returncode != 0 or not os.path.exists(wav_path):
        raise RuntimeError(f"ffmpeg audio extraction failed: {r.stderr.strip()[:200]}")


def _is_silent(wav_path: str, rms_floor: float = 0.003) -> bool:
    with wave.open(wav_path, "rb") as w:
        frames = w.readframes(w.getnframes())
    if not frames:
        return True
    a = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(a ** 2))) < rms_floor


_WHISPER = {}


def _load_wav_float32(wav_path: str) -> np.ndarray:
    """Read our 16 kHz mono PCM WAV into the float32 array Whisper accepts.

    Passing samples instead of a filename skips faster-whisper's own decoder,
    which calls av.open(..., metadata_errors=...) and breaks on newer PyAV
    releases (TypeError).  ffmpeg already produced a clean WAV, so this is safe.
    """
    with wave.open(wav_path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_whisper(wav_path: str, model_size: str):
    """Transcribe with faster-whisper (CTranslate2 build of openai/whisper).
    Raises if the package or model weights are unavailable."""
    from faster_whisper import WhisperModel

    if model_size not in _WHISPER:
        _WHISPER[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    audio = _load_wav_float32(wav_path)
    segs, _info = _WHISPER[model_size].transcribe(audio, language="en", vad_filter=True)
    return [Segment(text=normalize_text(s.text), start=float(s.start), end=float(s.end))
            for s in segs if s.text.strip()]


def _load_sidecar(path: str) -> Optional[List[Segment]]:
    """Offline fallback: '<video>.transcript.json' = [{start,end,text}, ...]."""
    side = os.path.splitext(path)[0] + ".transcript.json"
    if not os.path.exists(side):
        return None
    with open(side, encoding="utf-8") as fh:
        rows = json.load(fh)
    return [Segment(text=normalize_text(r["text"]), start=float(r["start"]),
                    end=float(r["end"])) for r in rows if r.get("text", "").strip()]


def extract_video(path: str, doc: ExtractedDocument,
                  whisper_model: str = "base") -> ExtractedDocument:
    if not _has_audio_stream(path):
        return doc.fail("MP4 has no audio track - nothing to transcribe")
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        extract_audio(path, wav)
        if _is_silent(wav):
            return doc.fail("MP4 audio track is silent - nothing to transcribe")
        segments, backend = None, None
        try:
            segments, backend = _transcribe_whisper(wav, whisper_model), f"whisper-{whisper_model}"
        except Exception as exc:  # package missing / weights not downloadable
            doc.warnings.append(
                f"Whisper unavailable ({type(exc).__name__}: {str(exc)[:200]}); "
                "trying sidecar transcript")
            side = _load_sidecar(path)
            if side:
                segments, backend = side, "sidecar-transcript (ASR NOT run)"
    if not segments:
        return doc.fail("No transcript could be produced for MP4 "
                        "(no speech detected or ASR model unavailable)")
    doc.segments = segments
    doc.meta["asr_backend"] = backend
    return doc


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def extract_file(path: str, whisper_model: str = "base") -> ExtractedDocument:
    """Extract one file. Never raises."""
    ext = os.path.splitext(path)[1].lower()
    doc = ExtractedDocument(doc_id=os.path.basename(path), path=path,
                            modality=SUPPORTED.get(ext, "unknown"))
    try:
        if ext not in SUPPORTED:
            return doc.fail(f"Unsupported file type '{ext}'")
        if not os.path.exists(path):
            return doc.fail("File not found")
        if os.path.getsize(path) == 0:
            return doc.fail("File is empty (0 bytes)")
        if ext == ".pdf":
            return extract_pdf(path, doc)
        if ext == ".docx":
            return extract_docx(path, doc)
        return extract_video(path, doc, whisper_model)
    except Exception as exc:  # noqa: BLE001 - the pipeline must never crash
        return doc.fail(f"Unreadable or corrupted {doc.modality.upper()}: "
                        f"{type(exc).__name__}: {str(exc)[:160]}")


def extract_text_file(path: str) -> ExtractedDocument:
    """Knowledge-base reference texts (.txt / .md)."""
    doc = ExtractedDocument(doc_id=os.path.basename(path), path=path, modality="reference")
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = normalize_text(fh.read())
        if text:
            doc.segments.append(Segment(text=text, page=1))
        else:
            doc.fail("empty reference file")
    except Exception as exc:  # noqa: BLE001
        doc.fail(f"cannot read reference: {exc}")
    return doc
