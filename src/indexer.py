"""Module 2 - chunking, embedding and FAISS vector indexing.

Two granularities are indexed per corpus:

* **chunks**   - 200-300-word windows with 50-word overlap (spec).  Used for the
  coarse "section level" view and metadata (page / timestamp range).
* **passages** - short sentence windows (>= 25 words, stride 1 sentence).  Used
  for exact, human-verifiable evidence and for coverage-based percentages.
  Short transcripts (a 15 s pitch is ~40 words) cannot be matched reliably
  against 250-word chunks, so passage-level matching is the primary signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .extractor import ExtractedDocument, fmt_range

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - numpy fallback keeps the engine usable
    faiss = None

# --------------------------------------------------------------------------- #
# sentence segmentation
# --------------------------------------------------------------------------- #
_ABBR = ["e.g.", "i.e.", "etc.", "vs.", "inc.", "ltd.", "co.", "dr.", "mr.", "mrs.",
         "ms.", "prof.", "u.s.", "u.k.", "fig.", "no.", "approx.", "al."]
_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def _is_heading(line: str) -> bool:
    words = line.split()
    if not words or len(words) > 8 or re.search(r"[.!?,;]$", line.strip()):
        return False
    if line.strip().endswith(":"):
        return True
    capped = sum(1 for w in words if w[0].isupper() or w[0].isdigit())
    return capped / len(words) >= 0.6


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []

    def flush():
        if not buf:
            return
        para = " ".join(buf)
        low = para
        for a in _ABBR:  # protect abbreviation periods from the splitter
            low = re.sub(re.escape(a), a.replace(".", "․"), low, flags=re.I)
        for s in _SPLIT.split(low):
            s = s.replace("․", ".").strip()
            if s:
                out.append(s)
        buf.clear()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            flush()
        elif _is_heading(line):
            flush()
            out.append(line)
        else:
            buf.append(line)
    flush()
    return out


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass
class Sentence:
    doc_id: str
    idx: int
    text: str
    page: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass
class Unit:
    """A chunk or passage: a run of consecutive sentences of one document."""
    uid: str
    doc_id: str
    kind: str                     # "chunk" | "passage" | "reference"
    text: str
    s_from: int                   # sentence index range [s_from, s_to]
    s_to: int
    pages: Tuple[Optional[int], Optional[int]] = (None, None)
    times: Tuple[Optional[float], Optional[float]] = (None, None)
    modality: str = ""
    n_words: int = 0
    boilerplate: bool = False

    def location(self) -> dict:
        loc: dict = {}
        if self.pages[0] is not None:
            loc["page"] = self.pages[0] if self.pages[0] == self.pages[1] else \
                f"{self.pages[0]}-{self.pages[1]}"
        if self.times[0] is not None:
            loc["timestamp"] = fmt_range(self.times[0], self.times[1])
        return loc


@dataclass
class DocIndexData:
    doc: ExtractedDocument
    all_sentences: List[Sentence] = field(default_factory=list)
    boiler_sentences: List[Sentence] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)
    chunks: List[Unit] = field(default_factory=list)
    passages: List[Unit] = field(default_factory=list)


def sentences_of(doc: ExtractedDocument) -> List[Sentence]:
    sents: List[Sentence] = []
    for seg in doc.segments:
        for s in split_sentences(seg.text):
            sents.append(Sentence(doc.doc_id, len(sents), s, seg.page, seg.start, seg.end))
    return sents


def _span_meta(sents: List[Sentence], a: int, b: int):
    pages = [s.page for s in sents[a:b + 1] if s.page is not None]
    st = [s.start for s in sents[a:b + 1] if s.start is not None]
    en = [s.end for s in sents[a:b + 1] if s.end is not None]
    return ((min(pages), max(pages)) if pages else (None, None),
            (min(st), max(en)) if st else (None, None))


def build_chunks(doc_id: str, modality: str, sents: List[Sentence],
                 chunk_words: int = 250, overlap_words: int = 50) -> List[Unit]:
    """Sentence-aligned 200-300-word chunks with ~50-word sliding overlap."""
    chunks: List[Unit] = []
    i, n = 0, len(sents)
    while i < n:
        j, w = i, 0
        while j < n and (w < chunk_words or j == i):
            w += sents[j].n_words
            j += 1
        a, b = i, j - 1
        pages, times = _span_meta(sents, a, b)
        chunks.append(Unit(f"{doc_id}#c{len(chunks)}", doc_id, "chunk",
                           " ".join(s.text for s in sents[a:b + 1]), a, b, pages, times,
                           modality, w))
        if j >= n:
            break
        # step back ~overlap_words so consecutive chunks share context
        back, k = 0, b
        while k > a and back < overlap_words:
            back += sents[k].n_words
            k -= 1
        i = max(k + 1, a + 1)
    return chunks


def build_passages(doc_id: str, modality: str, sents: List[Sentence],
                   min_words: int = 25, kind: str = "passage") -> List[Unit]:
    """Sliding sentence windows (stride 1) of >= `min_words` words."""
    out: List[Unit] = []
    seen = set()
    n = len(sents)
    for i in range(n):
        j, w = i, 0
        while j < n and w < min_words:
            w += sents[j].n_words
            j += 1
        a, b = i, j - 1
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pages, times = _span_meta(sents, a, b)
        out.append(Unit(f"{doc_id}#p{a}-{b}", doc_id, kind,
                        " ".join(s.text for s in sents[a:b + 1]), a, b, pages, times,
                        modality, w))
    return out


# --------------------------------------------------------------------------- #
# vector index
# --------------------------------------------------------------------------- #
class VectorIndex:
    """Exact cosine search: FAISS IndexFlatIP over L2-normalised vectors
    (inner product == cosine).  Falls back to a NumPy matmul if FAISS is absent."""

    def __init__(self, dim: int):
        self.dim = dim
        self.units: List[Unit] = []
        self._vecs: List[np.ndarray] = []
        self._index = faiss.IndexFlatIP(dim) if faiss is not None else None
        self._mat: Optional[np.ndarray] = None

    def add(self, units: List[Unit], vecs: np.ndarray) -> None:
        if not units:
            return
        vecs = np.ascontiguousarray(vecs.astype("float32"))
        self.units.extend(units)
        if self._index is not None:
            self._index.add(vecs)
        else:
            self._vecs.append(vecs)
            self._mat = None

    def __len__(self) -> int:
        return len(self.units)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = max(1, min(k, len(self.units)))
        queries = np.ascontiguousarray(queries.astype("float32"))
        if self._index is not None:
            return self._index.search(queries, k)
        if self._mat is None:
            self._mat = np.vstack(self._vecs)
        sims = queries @ self._mat.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx

    def search_excluding(self, vec: np.ndarray, exclude_doc: str, k: int):
        """Top-k neighbours NOT belonging to `exclude_doc` (self-match exclusion).
        Over-fetches then filters so same-document hits never crowd out results."""
        over = min(len(self.units), k + sum(1 for u in self.units if u.doc_id == exclude_doc))
        sims, idxs = self.search(vec[None, :], over)
        hits = [(self.units[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if i >= 0 and self.units[i].doc_id != exclude_doc]
        return hits[:k]


class Corpus:
    """Everything the detector needs: per-doc sentences/units + vector indexes."""

    def __init__(self, embedder, cfg):
        self.embedder = embedder
        self.cfg = cfg
        self.docs: Dict[str, DocIndexData] = {}
        self.chunk_index: Optional[VectorIndex] = None
        self.passage_index: Optional[VectorIndex] = None
        self.ref_index: Optional[VectorIndex] = None
        self.vec_of: Dict[str, np.ndarray] = {}

    def prepare(self, docs: List[ExtractedDocument]) -> None:
        from .boilerplate import find_boilerplate

        all_sents = {d.doc_id: sentences_of(d) for d in docs}
        flagged, self.boilerplate_info = find_boilerplate(all_sents, self.cfg)
        for d in docs:
            data = DocIndexData(d)
            data.all_sentences = all_sents[d.doc_id]
            drop = flagged.get(d.doc_id, set())
            data.boiler_sentences = [s for s in data.all_sentences if s.idx in drop]
            # effective sentences: boilerplate removed (zero weight), then re-indexed
            data.sentences = [Sentence(s.doc_id, i, s.text, s.page, s.start, s.end)
                              for i, s in enumerate(x for x in data.all_sentences
                                                    if x.idx not in drop)]
            data.chunks = build_chunks(d.doc_id, d.modality, data.sentences,
                                       self.cfg.chunk_words, self.cfg.chunk_overlap)
            data.passages = build_passages(d.doc_id, d.modality, data.sentences,
                                           self.cfg.passage_min_words)
            self.docs[d.doc_id] = data

    def build(self, refs: List[ExtractedDocument]) -> None:
        ref_units: List[Unit] = []
        self.ref_sents: Dict[str, List[Sentence]] = {}
        for r in refs:
            sents = sentences_of(r)
            self.ref_sents[r.doc_id] = sents
            ref_units += build_passages(r.doc_id, "reference", sents,
                                       self.cfg.passage_min_words, kind="reference")
        all_units = [u for d in self.docs.values() for u in d.chunks + d.passages] + ref_units
        self.embedder.fit([u.text for u in all_units])  # no-op for neural models
        dim = None
        for kind, attr in (("chunk", "chunk_index"), ("passage", "passage_index")):
            units = [u for d in self.docs.values()
                     for u in (d.chunks if kind == "chunk" else d.passages)]
            vecs = self.embedder.encode([u.text for u in units])
            dim = vecs.shape[1] if len(vecs) else dim
            if len(units):
                idx = VectorIndex(vecs.shape[1])
                idx.add(units, vecs)
                setattr(self, attr, idx)
            for u, v in zip(units, vecs):
                self.vec_of[u.uid] = v
        if ref_units:
            rv = self.embedder.encode([u.text for u in ref_units])
            self.ref_index = VectorIndex(rv.shape[1])
            self.ref_index.add(ref_units, rv)
