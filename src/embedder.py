"""Embedding back-ends.

* `SentenceTransformerEmbedder` - production default, all-MiniLM-L6-v2
  (384-d, L2-normalised so inner product == cosine).
* `TfidfEmbedder` - fully offline stand-in used ONLY when the MiniLM weights
  cannot be loaded.  It is lexical (stemmed word 1-2-grams), so it cannot
  recognise synonym-level paraphrase; every report states which back-end ran.
"""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

log = logging.getLogger("integrity.embedder")


class Embedder:
    name = "abstract"
    semantic = False          # True only for neural sentence embedders

    def fit(self, texts: List[str]) -> None:  # optional
        return None

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    semantic = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name.split("/")[-1]

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        v = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
        return v.astype("float32")


_STOP = set("""a an and are as at be been but by for from had has have he her his i in
is it its of on or our she so than that the their them then there these they this
those to was we were what when which who will with would you your""".split())


def _stem(w: str) -> str:
    for suf in ("ization", "isation", "ations", "ation", "ingly", "ings", "ing",
                "edly", "ies", "ied", "ive", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return w


def _analyzer(text: str) -> List[str]:
    toks = [_stem(t) for t in re.findall(r"[a-z0-9%]+", text.lower()) if t not in _STOP]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class TfidfEmbedder(Embedder):
    name = "tfidf-1-2gram (offline fallback)"

    def __init__(self, max_features: int = 6000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True,
                                   max_features=max_features, norm="l2")
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self.vec.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called first")
        if not texts:
            return np.zeros((0, len(self.vec.vocabulary_)), dtype="float32")
        return self.vec.transform(texts).toarray().astype("float32")


def load_embedder(kind: str = "auto",
                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return (embedder, notes).  `auto` tries MiniLM then falls back."""
    notes: List[str] = []
    if kind in ("auto", "minilm"):
        try:
            return SentenceTransformerEmbedder(model_name), notes
        except Exception as exc:  # noqa: BLE001
            msg = (f"Could not load {model_name} ({type(exc).__name__}: "
                   f"{str(exc)[:120]}). Falling back to offline TF-IDF embeddings - "
                   "semantic paraphrase detection is DEGRADED.")
            if kind == "minilm":
                raise RuntimeError(msg) from exc
            log.warning(msg)
            notes.append(msg)
    return TfidfEmbedder(), notes
