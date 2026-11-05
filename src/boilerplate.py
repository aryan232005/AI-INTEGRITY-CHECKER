"""Boilerplate / shared-template suppression using document frequency.

Idea: text that most submissions contain (an assignment prompt, a cover sheet)
is not evidence of copying.  A sentence found in >= 40 % of submissions is tagged
BOILERPLATE_PROMPT and gets zero weight.  Two practical refinements:

1. The test runs on *sentences* (normalised text key) before windows are built,
   because a shared prompt rarely aligns with chunk boundaries - the chunk that
   contains the prompt's last sentence also contains original text.
2. In small batches a prompt shared by 4 of 15 files (27 %) would slip under 40 %.
   A *relaxed* rule therefore also applies when >= `min_docs` documents share the
   sentence AND it sits at a consistent position (document head or tail, where
   prompts / citation blocks live).  This stops mid-document verbatim copying by
   a small group from being silently excused.  Both rules are configurable.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from .indexer import Sentence


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def find_boilerplate(sent_map: Dict[str, List[Sentence]], cfg):
    """Return ({doc_id: set(sentence idx)}, {key: info}) for boilerplate sentences."""
    n_docs = len(sent_map)
    if n_docs < 3:
        return {d: set() for d in sent_map}, {}
    occ: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for doc_id, sents in sent_map.items():
        total = max(1, len(sents))
        for s in sents:
            k = norm_key(s.text)
            if len(k.split()) >= 4:                       # ignore tiny fragments
                occ[k][doc_id].append(s.idx / total)
    flagged: Dict[str, Set[int]] = {d: set() for d in sent_map}
    boiler_keys: Dict[str, dict] = {}
    for k, per_doc in occ.items():
        df = len(per_doc) / n_docs
        positions = [p for ps in per_doc.values() for p in ps]
        head_tail = all(p <= 0.30 or p >= 0.85 for p in positions)
        # two files sharing a sentence is a copy, not a template: require >= min_docs
        strict = df >= cfg.boilerplate_df and len(per_doc) >= cfg.boilerplate_min_docs
        relaxed = (len(per_doc) >= cfg.boilerplate_min_docs
                   and df >= cfg.boilerplate_relaxed_df and head_tail)
        if strict or relaxed:
            boiler_keys[k] = {"df": df, "docs": sorted(per_doc),
                              "rule": "strict-DF" if strict else "relaxed-DF+position"}
    for doc_id, sents in sent_map.items():
        for s in sents:
            if norm_key(s.text) in boiler_keys:
                flagged[doc_id].add(s.idx)
        # A document made (almost) entirely of shared text has no original content of its
        # own: that is duplication, not a template with an original answer.  Keep it scored.
        original_words = sum(s.n_words for s in sents if s.idx not in flagged[doc_id])
        if flagged[doc_id] and original_words < cfg.boilerplate_min_original_words:
            flagged[doc_id] = set()
    return flagged, boiler_keys
