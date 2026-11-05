#!/usr/bin/env python3
"""CLI entry point.

    python run_integrity_check.py --input_dir ./sample_submissions \
        --output_dir ./reports --knowledge_base ./knowledge_base
"""
import argparse
import logging
import os
import sys
import time
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from src import __version__
from src.config import Config
from src.detector import DocResult, IntegrityEngine, add_direction_hints
from src.embedder import load_embedder
from src.extractor import SUPPORTED, extract_file, extract_text_file
from src.indexer import Corpus
from src.reporter import write_reports, write_summary

console = Console()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multimodal submission integrity checker")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--knowledge_base", default=None,
                   help="folder of .txt/.md reference sources for external matching")
    p.add_argument("--embedder", choices=["auto", "minilm", "tfidf"], default="auto")
    p.add_argument("--ai_backend", choices=["heuristic", "roberta"], default="heuristic",
                   help="roberta = HF openai-community/roberta-base-openai-detector (opt-in download)")
    p.add_argument("--whisper_model", default="base", help="tiny | base | path")
    p.add_argument("--threshold", type=float, default=None, help="override match threshold")
    p.add_argument("--boilerplate_df", type=float, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = Config(embedder=a.embedder, ai_backend=a.ai_backend, whisper_model=a.whisper_model)
    if a.boilerplate_df is not None:
        cfg.boilerplate_df = a.boilerplate_df
    if not os.path.isdir(a.input_dir):
        console.print(f"[red]Input directory not found: {a.input_dir}[/red]")
        return 2
    files = sorted(f for f in os.listdir(a.input_dir)
                   if os.path.splitext(f)[1].lower() in SUPPORTED)
    if not files:
        console.print("[red]No .pdf/.docx/.mp4 files found.[/red]")
        return 2
    console.print(f"[bold]Integrity checker v{__version__}[/bold] - {len(files)} submissions")
    timings, t0 = {}, time.time()

    # 1. ingestion --------------------------------------------------------
    docs = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TimeElapsedColumn(), console=console) as pg:
        task = pg.add_task("Ingesting", total=len(files))
        for f in files:
            pg.update(task, description=f"Ingesting {f}")
            d = extract_file(os.path.join(a.input_dir, f), cfg.whisper_model)
            docs.append(d)
            mark = "[green]ok[/green]" if d.ok else f"[yellow]SKIPPED: {d.error}[/yellow]"
            pg.console.print(f"  {f:<34} {d.modality:<6} {d.n_words:>5} words  {mark}")
            pg.advance(task)
    timings["ingestion"] = round(time.time() - t0, 2)
    good = [d for d in docs if d.ok]
    refs = []
    if a.knowledge_base and os.path.isdir(a.knowledge_base):
        for f in sorted(os.listdir(a.knowledge_base)):
            if f.lower().endswith((".txt", ".md")):
                r = extract_text_file(os.path.join(a.knowledge_base, f))
                if r.ok:
                    refs.append(r)
        console.print(f"Knowledge base: {len(refs)} reference documents")
    if not good:
        console.print("[red]No processable documents.[/red]")

    # 2. indexing ---------------------------------------------------------
    t1 = time.time()
    embedder, notes = load_embedder(cfg.embedder, cfg.model_name)
    for n in notes:
        console.print(f"[yellow]WARNING: {n}[/yellow]")
    for d in good:
        if "sidecar" in str(d.meta.get("asr_backend", "")):
            notes.append(f"{d.doc_id}: transcript came from a sidecar file - Whisper was "
                         "not available in this environment, so ASR was NOT exercised.")
    corpus = Corpus(embedder, cfg)
    corpus.prepare(good)
    corpus.build(refs)
    timings["indexing"] = round(time.time() - t1, 2)
    n_p = len(corpus.passage_index) if corpus.passage_index else 0
    n_c = len(corpus.chunk_index) if corpus.chunk_index else 0
    console.print(f"Indexed {n_c} chunks / {n_p} passages with {embedder.name} "
                  f"in {timings['indexing']}s")
    if corpus.boilerplate_info:
        console.print(f"Boilerplate filter: {len(corpus.boilerplate_info)} shared template "
                      "sentence(s) excluded from scoring")

    # 3. scoring ----------------------------------------------------------
    t2 = time.time()
    engine = IntegrityEngine(corpus, cfg)
    if a.threshold is not None:
        engine.thr = a.threshold
    results, matrix = [], defaultdict(dict)
    for d in docs:
        if not d.ok:
            results.append(DocResult(d.doc_id, d.modality, status="UNPROCESSABLE",
                                     warnings=list(d.warnings), risk="UNPROCESSABLE"))
            continue
        r = engine.analyse(d.doc_id)
        results.append(r)
        for other, pct in r.coverage_by_doc.items():
            matrix[d.doc_id][other] = pct
    add_direction_hints(results)
    timings["scoring"] = round(time.time() - t2, 2)

    # 4. reporting --------------------------------------------------------
    run_info = {"embedding_backend": embedder.name, "match_threshold": engine.thr,
                "direct_threshold": engine.direct, "version": __version__,
                "notes": notes, "n_documents": len(docs)}
    write_reports(results, a.output_dir, run_info)
    timings["total"] = round(time.time() - t0, 2)
    write_summary(results, matrix, a.output_dir, run_info, timings)

    tbl = Table(title="Results")
    for c in ("Document", "Risk", "Peer %", "External %", "AI %"):
        tbl.add_column(c)
    colour = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green", "UNPROCESSABLE": "magenta"}
    for r in results:
        sc = r.scores or {}
        ai = sc.get("ai_generated_probability")
        tbl.add_row(r.doc_id, f"[{colour[r.risk]}]{r.risk}[/]",
                    str(sc.get("peer_similarity_percentage", "-")),
                    str(sc.get("external_similarity_percentage", "-")),
                    "n/a" if ai is None and sc else str(ai if ai is not None else "-"))
    console.print(tbl)
    console.print(f"Reports written to [bold]{a.output_dir}[/bold] "
                  f"(per-document .json/.md + summary.md) in {timings['total']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
