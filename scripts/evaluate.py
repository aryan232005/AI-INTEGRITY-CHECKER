#!/usr/bin/env python3
"""Score the pipeline against sample_submissions/dataset_manifest.json.

    python run_integrity_check.py --input_dir sample_submissions --output_dir reports --knowledge_base knowledge_base
    python scripts/evaluate.py --reports reports --manifest sample_submissions/dataset_manifest.json

Prints Markdown tables (paste into EVALUATION.md) and writes evaluation_metrics.json.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAIR_FLAG_PCT = 15.0      # pair flagged if either doc has >=15% of its words in the other
EXT_FLAG_PCT = 15.0
AI_FLAG = 70.0


def confusion(gold, pred):
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec, f1=f1)


def table(name, c):
    return (f"**{name}**\n\n| | Predicted flagged | Predicted clean |\n|---|---|---|\n"
            f"| **Actually positive** | TP = {c['TP']} | FN = {c['FN']} |\n"
            f"| **Actually negative** | FP = {c['FP']} | TN = {c['TN']} |\n\n"
            f"Precision = {c['precision']:.2f}, Recall = {c['recall']:.2f}, F1 = {c['f1']:.2f}\n")


def cosine_distribution(input_dir, kb_dir, gold_pairs):
    """Best passage-cosine for each doc pair using the same embedder as the pipeline."""
    from src.config import Config
    from src.embedder import load_embedder
    from src.extractor import SUPPORTED, extract_file
    from src.indexer import Corpus

    cfg = Config()
    docs = [extract_file(os.path.join(input_dir, f)) for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in SUPPORTED]
    good = [d for d in docs if d.ok]
    emb, _ = load_embedder("auto", cfg.model_name)
    c = Corpus(emb, cfg)
    c.prepare(good)
    c.build([])
    vecs = {d.doc_id: np.array([c.vec_of[u.uid] for u in c.docs[d.doc_id].passages])
            for d in good if c.docs[d.doc_id].passages}
    out = {}
    for a, b in itertools.combinations(sorted(vecs), 2):
        out[(a, b)] = float((vecs[a] @ vecs[b].T).max())
    pos = [v for k, v in out.items() if frozenset(k) in gold_pairs]
    neg = [v for k, v in out.items() if frozenset(k) not in gold_pairs]
    return emb.name, pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--manifest", default="sample_submissions/dataset_manifest.json")
    ap.add_argument("--input_dir", default="sample_submissions")
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.reports, "summary.json")))
    man = json.load(open(a.manifest))
    docs = {d["document_id"]: d for d in summ["documents"]}
    ok = sorted(k for k, d in docs.items() if d["risk"] != "UNPROCESSABLE")
    M = summ["peer_overlap_matrix_pct"]
    gold_pairs = {frozenset((p["a"], p["b"])) for p in man["peer_overlap_pairs"]}

    # 1. pair-level overlap detection --------------------------------------
    pairs = list(itertools.combinations(ok, 2))
    gold = [frozenset(p) in gold_pairs for p in pairs]
    pred = [max(M.get(x, {}).get(y, 0), M.get(y, {}).get(x, 0)) >= PAIR_FLAG_PCT
            for x, y in pairs]
    c_pair = confusion(gold, pred)
    missed = [p for p, g, q in zip(pairs, gold, pred) if g and not q]
    spurious = [p for p, g, q in zip(pairs, gold, pred) if q and not g]
    # per-type recall
    by_type = {}
    for p in man["peer_overlap_pairs"]:
        k = (p["a"], p["b"])
        hit = pred[pairs.index(k)] if k in pairs else (pred[pairs.index(k[::-1])]
                                                       if k[::-1] in pairs else False)
        by_type.setdefault(p["type"], []).append(hit)

    # 2. external ------------------------------------------------------------
    g_ext = [d in man["external_plagiarism"] for d in ok]
    p_ext = [(docs[d]["scores"]["external_similarity_percentage"] or 0) >= EXT_FLAG_PCT for d in ok]
    c_ext = confusion(g_ext, p_ext)

    # 3. AI (text documents only; ASR transcripts are n/a by design) -----------
    txt = [d for d in ok if docs[d]["scores"]["ai_generated_probability"] is not None]
    g_ai = [d in man["ai_generated"] for d in txt]
    p_ai = [docs[d]["scores"]["ai_generated_probability"] >= AI_FLAG for d in txt]
    c_ai = confusion(g_ai, p_ai)

    # 4. boilerplate false positives ------------------------------------------
    bp = man["boilerplate_group"]
    bp_pairs = list(itertools.combinations(bp, 2))
    bp_fp = sum(max(M.get(x, {}).get(y, 0), M.get(y, {}).get(x, 0)) >= PAIR_FLAG_PCT
                for x, y in bp_pairs)

    # 5. robustness -------------------------------------------------------------
    unproc = sorted(k for k, d in docs.items() if d["risk"] == "UNPROCESSABLE")
    robust_ok = unproc == sorted(man["unprocessable"])

    emb_name, pos, neg = cosine_distribution(a.input_dir, None, gold_pairs)
    md = [f"Embedding back-end used for this evaluation: `{emb_name}`", "",
          table(f"Pair-level overlap detection ({len(pairs)} document pairs, "
                f"flag if >= {PAIR_FLAG_PCT:.0f}% overlap)", c_pair)]
    md.append("Recall by relationship type: " + ", ".join(
        f"{t}: {sum(v)}/{len(v)}" for t, v in by_type.items()) + "\n")
    md.append(table(f"External-source plagiarism ({len(ok)} documents, flag >= {EXT_FLAG_PCT:.0f}%)", c_ext))
    md.append(table(f"AI-generated detection ({len(txt)} text documents, flag >= {AI_FLAG:.0f}%)", c_ai))
    md.append(f"**Boilerplate (Scenario 4):** {bp_fp} of {len(bp_pairs)} template-sharing "
              f"pairs falsely flagged.\n")
    md.append(f"**Robustness (Scenario 3):** unprocessable files = {unproc}; "
              f"matches manifest: {robust_ok}; pipeline completed without crashing.\n")
    md.append("**Best passage-cosine per document pair**\n")
    md.append(f"| Group | n | min | mean | max |\n|---|---|---|---|---|\n"
              f"| True overlap pairs | {len(pos)} | {min(pos):.3f} | {np.mean(pos):.3f} | {max(pos):.3f} |\n"
              f"| Unrelated pairs | {len(neg)} | {min(neg):.3f} | {np.mean(neg):.3f} | {max(neg):.3f} |\n")
    print("\n".join(md))
    print("Missed pairs:", missed, "\nSpurious pairs:", spurious)
    json.dump({"embedding": emb_name, "pair": c_pair, "external": c_ext, "ai": c_ai,
               "boilerplate_false_pairs": bp_fp, "missed_pairs": [list(m) for m in missed],
               "spurious_pairs": [list(s) for s in spurious],
               "cosine_positive": pos, "cosine_negative_max": max(neg)},
              open(os.path.join(a.reports, "evaluation_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
