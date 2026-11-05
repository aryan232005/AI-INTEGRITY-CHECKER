"""Module 4 - explainable reports (JSON + Markdown) and the batch summary."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from .detector import DocResult

_BADGE = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]",
          "UNPROCESSABLE": "[UNPROCESSABLE]"}


def _loc(loc: dict, side: str) -> str:
    bits = []
    for key, label in ((f"{side}_page", "page"), (f"{side}_timestamp", "time")):
        if key in loc:
            est = " (est.)" if loc.get(f"{side}_page_estimated") and key.endswith("page") else ""
            bits.append(f"{label} {loc[key]}{est}")
    return ", ".join(bits) or "n/a"


def to_json(r: DocResult, run_info: dict) -> dict:
    return {
        "document_id": r.doc_id,
        "total_integrity_risk": r.risk if r.status == "OK" else "UNPROCESSABLE",
        "status": r.status,
        "scores": r.scores,
        "peer_matches": r.peer_matches,
        "multimodal_matches": r.multimodal_matches,
        "external_matches": r.external_matches,
        "explanations": r.explanation,
        "direction_hints": r.direction_hints,
        "warnings": r.warnings,
        "boilerplate_excluded": r.boilerplate,
        "section_level_view": r.chunk_level,
        "ai_detection_detail": r.ai_detail,
        "coverage_by_document_pct": r.coverage_by_doc,
        "run_info": run_info,
    }


def _match_block(i: int, m: dict, kind: str) -> List[str]:
    loc = m["location"]
    L = [f"**{i}. {m['matched_file']}** - cosine **{m['similarity_score']}** "
         f"({m['match_type']}, word-overlap Jaccard {m['lexical_jaccard']})", ""]
    L.append(f"- This document: {_loc(loc, 'query')}")
    if kind == "external":
        L.append(f"- Reference: {m['matched_file']} ({loc.get('reference_passage', '')})")
    else:
        L.append(f"- Matched document: {_loc(loc, 'matched')}")
    L += ["", f"> **Submitted:** {m['query_passage']}", ">",
          f"> **Source:** {m['matched_passage']}", ""]
    return L


def to_markdown(r: DocResult, run_info: dict) -> str:
    risk = r.risk if r.status == "OK" else "UNPROCESSABLE"
    L = [f"# Integrity report - `{r.doc_id}`", "",
         f"**Risk: {risk}** | type: {r.modality} | words analysed: {r.n_words}", ""]
    if r.status != "OK":
        L += ["## Could not be processed", ""] + [f"- {w}" for w in r.warnings]
        L += ["", "_The file was skipped; the rest of the batch was unaffected. "
                  "A human should obtain a valid copy._", ""]
        return "\n".join(L)
    s = r.scores
    ai = s["ai_generated_probability"]
    L += ["## Scores", "", "| Signal | Value |", "|---|---|",
          f"| Internal peer similarity | {s['peer_similarity_percentage']}% |",
          f"| External (reference) match | {s['external_similarity_percentage']}% |",
          f"| AI-generated likelihood | {'n/a' if ai is None else str(ai) + '%'} |", "",
          "## Why", "", r.explanation, ""]
    if r.coverage_by_doc:
        L += ["## Overlap by document", "", "| Matched file | % of this document covered |",
              "|---|---|"] + [f"| {d} | {p}% |" for d, p in r.coverage_by_doc.items()] + [""]
    for title, items, kind in (("Peer matches", r.peer_matches, "peer"),
                               ("Cross-modal matches (video <-> document)",
                                r.multimodal_matches, "peer"),
                               ("External reference matches", r.external_matches, "external")):
        if items:
            L += [f"## {title}", ""]
            for i, m in enumerate(items, 1):
                L += _match_block(i, m, kind)
    if r.boilerplate:
        L += ["## Excluded as shared template / prompt (BOILERPLATE_PROMPT)", ""]
        for b in r.boilerplate[:6]:
            L.append(f"- \"{b['text'][:140]}{'...' if len(b['text']) > 140 else ''}\" "
                     f"(shared by {len(b['shared_by_documents'])} documents)")
        L.append("")
    if r.warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in r.warnings] + [""]
    L += ["---", f"_Embedding: {run_info['embedding_backend']} | thresholds: match >= "
          f"{run_info['match_threshold']}, direct >= {run_info['direct_threshold']} | "
          f"AI backend: {r.ai_detail.get('backend', 'n/a')}_"]
    return "\n".join(L)


def write_reports(results: List[DocResult], out_dir: str, run_info: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        stem = os.path.join(out_dir, r.doc_id)
        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump(to_json(r, run_info), fh, indent=2, ensure_ascii=False)
        with open(stem + ".md", "w", encoding="utf-8") as fh:
            fh.write(to_markdown(r, run_info))


def write_summary(results: List[DocResult], matrix: Dict[str, Dict[str, float]],
                  out_dir: str, run_info: dict, timings: dict) -> None:
    ok = [r for r in results if r.status == "OK"]
    ids = [r.doc_id for r in ok]
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_info": run_info, "timings_s": timings,
                   "documents": [{"document_id": r.doc_id,
                                  "risk": r.risk if r.status == "OK" else "UNPROCESSABLE",
                                  "scores": r.scores, "warnings": r.warnings}
                                 for r in results],
                   "peer_overlap_matrix_pct": matrix}, fh, indent=2)
    with open(os.path.join(out_dir, "similarity_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["query \\ matched"] + ids)
        for a in ids:
            w.writerow([a] + [matrix.get(a, {}).get(b, 0.0) if a != b else "" for b in ids])
    L = ["# Batch summary", "",
         f"Documents: {len(results)} ({len(ok)} processed, {len(results) - len(ok)} skipped) | "
         f"embedding: **{run_info['embedding_backend']}** | "
         f"match threshold {run_info['match_threshold']}", ""]
    for n in run_info.get("notes", []):
        L += [f"> NOTE: {n}", ""]
    L += ["## Risk overview", "", "| Document | Risk | Peer % | External % | AI % | Top overlap |",
          "|---|---|---|---|---|---|"]
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNPROCESSABLE": 3}
    for r in sorted(results, key=lambda r: (order[r.risk if r.status == "OK" else "UNPROCESSABLE"], r.doc_id)):
        if r.status != "OK":
            L.append(f"| {r.doc_id} | UNPROCESSABLE | - | - | - | {r.warnings[-1]} |")
            continue
        top = next(iter(r.coverage_by_doc.items()), None)
        if top is None and r.external_matches:
            top = (r.external_matches[0]["matched_file"] + " [ref]",
                   r.scores["external_similarity_percentage"])
        ai = r.scores["ai_generated_probability"]
        L.append(f"| {r.doc_id} | {r.risk} | {r.scores['peer_similarity_percentage']} | "
                 f"{r.scores['external_similarity_percentage']} | "
                 f"{'n/a' if ai is None else ai} | "
                 f"{(top[0] + ' (' + str(top[1]) + '%)') if top else '-'} |")
    L += ["", "## Peer overlap matrix (% of ROW document's words matched in COLUMN document)", ""]
    short = {d: d[:2] if d[:2].isdigit() else d[:6] for d in ids}
    L.append("| |" + "|".join(short[d] for d in ids) + "|")
    L.append("|---|" + "---|" * len(ids))
    for a in ids:
        row = [("-" if a == b else (f"{matrix.get(a, {}).get(b, 0):.0f}" if matrix.get(a, {}).get(b) else "."))
               for b in ids]
        L.append(f"| {short[a]} {a} |" + "|".join(row) + "|")
    L += ["", f"Timings (s): {timings}"]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
