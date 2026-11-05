"""Module 3 - multi-signal integrity engine.

Signals
  1. internal peer similarity  (text<->text and video<->text, self-matches excluded)
  2. external match            (local reference knowledge base)
  3. AI-generated likelihood   (RoBERTa detector if loadable, else documented heuristics)

Percentages are *coverage* figures: the share of a document's non-boilerplate
words that sit inside a window whose best match clears the similarity threshold.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .boilerplate import norm_key
from .indexer import Corpus, Sentence, Unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingle_jaccard(a: str, b: str, n: int = 3) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) < n or len(tb) < n:
        n = 1
    sa = {tuple(ta[i:i + n]) for i in range(len(ta) - n + 1)}
    sb = {tuple(tb[i:i + n]) for i in range(len(tb) - n + 1)}
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


@dataclass
class Match:
    query: Unit
    target: Unit
    score: float
    jaccard: float


@dataclass
class DocResult:
    doc_id: str
    modality: str
    status: str = "OK"                      # OK | UNPROCESSABLE
    warnings: List[str] = field(default_factory=list)
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    peer_matches: List[dict] = field(default_factory=list)
    multimodal_matches: List[dict] = field(default_factory=list)
    external_matches: List[dict] = field(default_factory=list)
    coverage_by_doc: Dict[str, float] = field(default_factory=dict)
    boilerplate: List[dict] = field(default_factory=list)
    chunk_level: List[dict] = field(default_factory=list)
    ai_detail: dict = field(default_factory=dict)
    direction_hints: List[str] = field(default_factory=list)
    risk: str = "LOW"
    explanation: str = ""
    n_words: int = 0
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# AI-generated text estimation
# --------------------------------------------------------------------------- #
_LLM_PHRASES = [
    r"\bmoreover\b", r"\bfurthermore\b", r"\badditionally\b", r"\bin conclusion\b",
    r"\bin summary\b", r"\boverall\b", r"\bit is (important|crucial|essential) to\b",
    r"\bplays? a (crucial|pivotal|vital|key) role\b", r"\bdelve", r"\blandscape\b",
    r"\bmultifaceted\b", r"\bholistic", r"\brobust\b", r"\bseamless", r"\bpivotal\b",
    r"\bfoster(s|ing)?\b", r"\bunderscor", r"\bnavigat", r"\bever-(evolving|changing)\b",
    r"\bin today'?s\b", r"\bleverag", r"\bstakeholders?\b", r"\bcomprehensive\b",
    r"\bsynerg", r"\btestament\b", r"\bparadigm\b", r"\bnot only\b.{0,80}\bbut also\b",
    r"\bensur(e|es|ing)\b", r"\bcrucial\b", r"\bkey takeaway", r"\bsignificantly\b",
]
_HUMAN_CUES = [r"\bI\b", r"\bwe\b", r"\bmy\b", r"\bour\b", r"n't\b", r"\bI'm\b", r"\b\w+'s\b",
               r"\(\s*\d", r"\bbut\b", r"\bhowever\b"]


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 3]


def heuristic_ai_score(text: str) -> Tuple[Optional[float], dict]:
    """Reproducible stylometric estimator (NOT a trained detector).

    Features (LLM prose tends to be): uniform sentence length (low burstiness),
    dense with stock connectives/phrases, low on first-person voice and concrete
    digits, and highly regular in paragraph-opening patterns.

    The weights below are HAND-CHOSEN, not fitted to data: each is a guess at how
    strongly the feature should move the score, set before seeing the sample
    results.  The centring constants (0.55, 1.5, 1.0, 0.25) are my rough "typical
    human text" values.  They are a starting point for calibration on labelled
    data, not a validated model - see EVALUATION.md, failure case 3.
    """
    words = _tokens(text)
    sents = _sentences(text)
    if len(words) < 80 or len(sents) < 4:
        return None, {"reason": "insufficient text (<80 words / <4 sentences)"}
    lens = [len(s.split()) for s in sents]
    mean_len = statistics.mean(lens)
    cv = statistics.pstdev(lens) / mean_len if mean_len else 0.0     # burstiness
    phrase_hits = sum(len(re.findall(p, text, flags=re.I)) for p in _LLM_PHRASES)
    phrase_density = 100.0 * phrase_hits / len(words)                  # per 100 words
    human_hits = sum(len(re.findall(p, text)) for p in _HUMAN_CUES[:6])
    human_density = 100.0 * human_hits / len(words)
    digit_density = 100.0 * sum(c.isdigit() for c in text) / max(1, len(text)) * 10
    starts = [s.split()[0].lower() for s in sents]
    start_repeat = 1 - len(set(starts)) / len(starts)
    z = (-0.4
         + 3.0 * (0.55 - cv) / 0.25            # low burstiness -> AI
         + 0.9 * (phrase_density - 1.5)        # stock phrases -> AI
         - 0.7 * (human_density - 1.0)         # first-person voice -> human
         - 0.25 * (digit_density - 1.0)        # concrete figures -> human
         + 2.0 * (start_repeat - 0.25))
    prob = 100.0 / (1.0 + math.exp(-z / 2.0))
    return round(prob, 1), {
        "backend": "stylometric-heuristic",
        "sentence_length_cv": round(cv, 3), "mean_sentence_words": round(mean_len, 1),
        "llm_phrase_per_100w": round(phrase_density, 2),
        "first_person_per_100w": round(human_density, 2),
        "digit_signal": round(digit_density, 2),
        "repeated_sentence_openers": round(start_repeat, 3), "logit": round(z, 2)}


_ROBERTA = {}


def roberta_ai_score(text: str, model_name: str) -> Tuple[Optional[float], dict]:
    """Average 'fake' probability of an HF RoBERTa detector over 300-word blocks."""
    from transformers import pipeline

    if model_name not in _ROBERTA:
        _ROBERTA[model_name] = pipeline("text-classification", model=model_name,
                                        truncation=True, max_length=512, top_k=None)
    pipe = _ROBERTA[model_name]
    words = text.split()
    if len(words) < 40:
        return None, {"reason": "insufficient text"}
    blocks = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
    probs = []
    for out in pipe(blocks):
        d = {o["label"].lower(): o["score"] for o in out}
        probs.append(d.get("fake", d.get("ai", d.get("label_1", 0.0))))
    score = round(100 * sum(probs) / len(probs), 1)
    return score, {"backend": model_name, "blocks": len(blocks)}


def estimate_ai(text: str, backend: str = "heuristic") -> Tuple[Optional[float], dict]:
    """`heuristic` (default, reproducible, no download) or `roberta` (opt-in).

    The RoBERTa-base OpenAI detector was trained on GPT-2 output and is known to
    be weak on modern LLM text, so it is not the default."""
    if backend == "roberta":
        try:
            return roberta_ai_score(text, "openai-community/roberta-base-openai-detector")
        except Exception as exc:  # noqa: BLE001
            score, detail = heuristic_ai_score(text)
            detail["fallback_reason"] = f"roberta backend unavailable ({type(exc).__name__})"
            return score, detail
    return heuristic_ai_score(text)


def add_direction_hints(results: List["DocResult"], min_gap: float = 40.0) -> None:
    """Containment asymmetry as a weak hint about which file is the derived one.

    If 100% of B sits inside A but only 25% of A sits inside B, B is probably
    built from A (or from a shared source).  This is only a hint: it also fires
    when a short document is quoted inside a long one, and it says nothing about
    who wrote first, so reports say "check submission times".
    """
    by_id = {r.doc_id: r for r in results if r.status == "OK"}
    for r in by_id.values():
        for other_id, mine in r.coverage_by_doc.items():
            other = by_id.get(other_id)
            if other is None:
                continue
            theirs = other.coverage_by_doc.get(r.doc_id, 0.0)
            if mine - theirs >= min_gap:
                r.direction_hints.append(
                    f"{mine:.0f}% of this file appears in {other_id}, but only "
                    f"{theirs:.0f}% of {other_id} appears here: this file looks derived "
                    f"from (or a subset of) {other_id}.")
        if r.direction_hints:
            r.explanation += " " + " ".join(r.direction_hints)


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class IntegrityEngine:
    def __init__(self, corpus: Corpus, cfg):
        self.corpus, self.cfg = corpus, cfg
        sem = getattr(corpus.embedder, "semantic", False)
        self.thr = cfg.match_threshold if sem else cfg.fallback_match_threshold
        self.direct = cfg.direct_threshold if sem else cfg.fallback_direct_threshold

    # ---- matching ---------------------------------------------------------
    def _hits(self, doc_id: str, external: bool) -> List[Match]:
        data = self.corpus.docs[doc_id]
        index = self.corpus.ref_index if external else self.corpus.passage_index
        out: List[Match] = []
        if index is None or not len(index):
            return out
        for u in data.passages:
            vec = self.corpus.vec_of[u.uid]
            if external:
                sims, idxs = index.search(vec[None, :], self.cfg.top_k)
                hits = [(index.units[i], float(s)) for s, i in zip(sims[0], idxs[0]) if i >= 0]
            else:
                hits = index.search_excluding(vec, doc_id, self.cfg.top_k)  # no self-matches
            best: Dict[str, Match] = {}
            for t, s in hits:
                if s >= self.thr and (t.doc_id not in best or s > best[t.doc_id].score):
                    best[t.doc_id] = Match(u, t, s, shingle_jaccard(u.text, t.text))
            out += best.values()
        return out

    @staticmethod
    def _coverage(sents: List[Sentence], matches: List[Match]) -> Tuple[float, Dict[str, float]]:
        total = sum(s.n_words for s in sents)
        if not total:
            return 0.0, {}
        covered, by_doc = set(), defaultdict(set)
        for m in matches:
            for i in range(m.query.s_from, m.query.s_to + 1):
                covered.add(i)
                by_doc[m.target.doc_id].add(i)

        def w(idx):
            return sum(sents[i].n_words for i in idx)

        return (100.0 * w(covered) / total,
                {d: round(100.0 * w(ix) / total, 1) for d, ix in by_doc.items()})

    def _evidence(self, matches: List[Match], q_doc, limit: int, external=False) -> List[dict]:
        """Best match per (query window, target doc), overlap-pruned, top-`limit`."""
        kept: List[Match] = []
        for m in sorted(matches, key=lambda m: -m.score):
            dup = False
            for k in kept:
                if k.target.doc_id != m.target.doc_id:
                    continue
                lo = max(k.query.s_from, m.query.s_from)
                hi = min(k.query.s_to, m.query.s_to)
                span = min(k.query.s_to - k.query.s_from, m.query.s_to - m.query.s_from) + 1
                if hi >= lo and (hi - lo + 1) / span >= 0.5:
                    dup = True
                    break
            if not dup:
                kept.append(m)
        return [self._fmt(m, q_doc, external) for m in kept[:limit]]

    def _fmt(self, m: Match, q_doc: str, external: bool) -> dict:
        ql, tl = m.query.location(), m.target.location()
        loc = {}
        if "page" in ql:
            loc["query_page"] = ql["page"]
            if self.corpus.docs[q_doc].doc.meta.get("page_numbers_estimated"):
                loc["query_page_estimated"] = True
        if "timestamp" in ql:
            loc["query_timestamp"] = ql["timestamp"]
        if not external:
            if "page" in tl:
                loc["matched_page"] = tl["page"]
                if self.corpus.docs[m.target.doc_id].doc.meta.get("page_numbers_estimated"):
                    loc["matched_page_estimated"] = True
            if "timestamp" in tl:
                loc["matched_timestamp"] = tl["timestamp"]
        else:
            loc["reference_passage"] = m.target.uid.split("#")[-1]
        return {
            "matched_file": m.target.doc_id,
            "similarity_score": round(m.score, 3),
            "match_type": "DIRECT_COPY" if m.score >= self.direct else "PARAPHRASE",
            "lexical_jaccard": round(m.jaccard, 3),
            "query_passage": m.query.text,
            "matched_passage": m.target.text,
            "location": loc,
        }

    # ---- per-document -----------------------------------------------------
    def analyse(self, doc_id: str) -> DocResult:
        data = self.corpus.docs[doc_id]
        d = data.doc
        res = DocResult(doc_id, d.modality, warnings=list(d.warnings), n_words=d.n_words,
                        meta=dict(d.meta))
        res.boilerplate = [{"tag": "BOILERPLATE_PROMPT", "text": s.text,
                            "location": {"page": s.page} if s.page else {},
                            "shared_by_documents": self.corpus.boilerplate_info.get(
                                norm_key(s.text), {}).get("docs", [])}
                           for s in data.boiler_sentences]
        if not data.sentences:
            res.warnings.append("No original (non-boilerplate) content left to score.")
            res.scores = {"peer_similarity_percentage": 0.0,
                          "external_similarity_percentage": 0.0,
                          "ai_generated_probability": None}
            return res
        # peer + cross-modal
        peer_hits = self._hits(doc_id, external=False)
        peer_pct, cov = self._coverage(data.sentences, peer_hits)
        same, cross = [], []
        for m in peer_hits:
            (same if m.target.modality == d.modality or
             {m.target.modality, d.modality} <= {"pdf", "docx"} else cross).append(m)
        lim = self.cfg.max_matches_per_doc
        res.peer_matches = self._evidence(same, doc_id, lim)
        res.multimodal_matches = self._evidence(cross, doc_id, lim)
        res.coverage_by_doc = dict(sorted(cov.items(), key=lambda kv: -kv[1]))
        # external
        ext_hits = self._hits(doc_id, external=True)
        ext_pct, ext_cov = self._coverage(data.sentences, ext_hits)
        res.external_matches = self._evidence(ext_hits, doc_id, lim, external=True)
        # AI
        text = " ".join(s.text for s in data.sentences)
        if d.modality == "video":
            ai, detail = None, {"reason": "not applicable to ASR transcripts "
                                          "(no reliable punctuation / sentence structure)"}
        else:
            ai, detail = estimate_ai(text, self.cfg.ai_backend)
        res.ai_detail = detail
        res.scores = {"peer_similarity_percentage": round(peer_pct, 1),
                      "external_similarity_percentage": round(ext_pct, 1),
                      "ai_generated_probability": ai}
        # coarse section-level view (chunk index)
        ci = self.corpus.chunk_index
        for c in data.chunks:
            hits = ci.search_excluding(self.corpus.vec_of[c.uid], doc_id, 1) if ci else []
            if hits and hits[0][1] >= self.thr:
                t, s = hits[0]
                res.chunk_level.append({"chunk": c.uid, "location": c.location(),
                                        "best_match_file": t.doc_id,
                                        "best_match_location": t.location(),
                                        "cosine": round(s, 3)})
        res.risk = self._risk(res)
        res.explanation = self._explain(res)
        return res

    def _risk(self, r: DocResult) -> str:
        ov = max(r.scores["peer_similarity_percentage"] or 0,
                 r.scores["external_similarity_percentage"] or 0)
        ai = r.scores["ai_generated_probability"] or 0
        ai_and_overlap = ai >= self.cfg.ai_high and ov >= self.cfg.medium_overlap_pct
        if ov >= self.cfg.high_overlap_pct or ai_and_overlap:
            return "HIGH"
        # AI likelihood alone never yields HIGH: detectors are unreliable evidence
        if ov >= self.cfg.medium_overlap_pct or ai >= self.cfg.ai_medium:
            return "MEDIUM"
        return "LOW"

    def _explain(self, r: DocResult) -> str:
        parts = []
        peer = r.scores["peer_similarity_percentage"]
        if peer:
            top = [(d, p) for d, p in r.coverage_by_doc.items()][:3]
            parts.append(
                f"{peer}% of the original content overlaps with other submissions "
                f"(" + ", ".join(f"{d}: {p}%" for d, p in top) + ").")
            pm = r.peer_matches + r.multimodal_matches
            if pm:
                kinds = {m["match_type"] for m in pm}
                lex = min(m["lexical_jaccard"] for m in pm)
                if "PARAPHRASE" in kinds and lex < 0.4:
                    parts.append("Some matches are paraphrases: high semantic similarity "
                                 "but low word-for-word overlap (Jaccard "
                                 f"as low as {lex}).")
            if r.multimodal_matches:
                parts.append("Cross-modal matches present (video transcript <-> document); "
                             "see multimodal_matches for timestamps and pages.")
            parts.append("Direction of copying cannot be determined from content alone - "
                         "check submission times.")
        ext = r.scores["external_similarity_percentage"]
        if ext:
            files = sorted({m["matched_file"] for m in r.external_matches})
            parts.append(f"{ext}% overlaps with reference sources ({', '.join(files)}).")
        ai = r.scores["ai_generated_probability"]
        if ai is not None and ai >= self.cfg.ai_medium:
            parts.append(f"AI-generation likelihood is {ai}% (backend: "
                         f"{r.ai_detail.get('backend')}); treat as a prompt for review, "
                         "not proof.")
        if r.boilerplate:
            parts.append(f"{len(r.boilerplate)} shared template/prompt sentence(s) were "
                         "tagged BOILERPLATE_PROMPT and excluded from scoring.")
        if not parts:
            parts.append("No significant overlap or AI-generation indicators found.")
        return " ".join(parts)
