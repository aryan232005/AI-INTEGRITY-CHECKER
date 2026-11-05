"""Central configuration: thresholds and tunables (all overridable via CLI)."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- chunking: ~250-word windows, 50-word overlap -------------------------
    chunk_words: int = 250
    chunk_overlap: int = 50
    # --- passage (sentence-window) units used for exact evidence -------------
    passage_min_words: int = 25
    # --- similarity thresholds (cosine) --------------------------------------
    # Defaults are for all-MiniLM-L6-v2; the offline TF-IDF/LSA fallback is
    # lexical, so its scale differs and gets its own defaults (see below).
    match_threshold: float = 0.75      # below this a hit is discarded
    direct_threshold: float = 0.85     # >= this: DIRECT copy, else PARAPHRASE
    fallback_match_threshold: float = 0.60
    fallback_direct_threshold: float = 0.80
    top_k: int = 5
    # --- boilerplate / shared-template suppression -----------------------------
    boilerplate_df: float = 0.40       # sentence present in >=40% of submissions
    boilerplate_min_docs: int = 3      # ...or in >=3 docs with the relaxed rule
    boilerplate_relaxed_df: float = 0.25
    boilerplate_min_original_words: int = 40   # fewer own words than this -> not a template
    boilerplate_sim: float = 0.90      # near-identical passage test
    # --- risk bands -----------------------------------------------------------
    high_overlap_pct: float = 40.0
    medium_overlap_pct: float = 15.0
    ai_high: float = 80.0
    ai_medium: float = 70.0
    # --- reporting ------------------------------------------------------------
    max_matches_per_doc: int = 12
    embedder: str = "auto"             # auto | minilm | tfidf
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    whisper_model: str = "base"
    ai_backend: str = "heuristic"      # heuristic | roberta
    extra: dict = field(default_factory=dict)
