#!/usr/bin/env python3
"""Rebuild ./sample_submissions, ./knowledge_base and dataset_manifest.json.

    python scripts/build_sample_dataset.py

Needs: reportlab, python-docx, ffmpeg, espeakng-loader (offline TTS).
Videos get a '<name>.transcript.json' sidecar with the exact sentence timings
used to build the audio; it is only read if Whisper cannot be loaded.
"""
import json
import os
import subprocess
import sys
import tempfile
import wave

import docx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, os.path.dirname(__file__))
import dataset_texts as T  # noqa: E402
from tts import synthesize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "sample_submissions")
KB = os.path.join(ROOT, "knowledge_base")
ST = getSampleStyleSheet()


def make_docx(name, blocks):
    d = docx.Document()
    for kind, text in blocks:
        d.add_heading(text, level=2) if kind == "h" else d.add_paragraph(text)
    d.save(os.path.join(OUT, name))


def make_pdf(name, pages, header=None, footer=True):
    """pages: list of list of (kind, text); one PDF page per list."""
    total = len(pages)

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        if header:
            canvas.drawString(50, A4[1] - 28, header)
        if footer:
            canvas.drawString(50, 22, f"Confidential - Page {doc.page} of {total}")
        canvas.restoreState()

    story = []
    for i, blocks in enumerate(pages):
        for kind, text in blocks:
            story.append(Paragraph(text, ST["Heading2"] if kind == "h" else ST["BodyText"]))
            story.append(Spacer(1, 6))
        if i < total - 1:
            story.append(PageBreak())
    SimpleDocTemplate(os.path.join(OUT, name), pagesize=A4, topMargin=60,
                      bottomMargin=50).build(story, onFirstPage=deco, onLaterPages=deco)


def make_video(name, sentences, silent=False, seconds=12):
    path = os.path.join(OUT, name)
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=c=0x1f3a5f:s=640x360:r=10"]
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    if silent:
        subprocess.run(base + ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t",
                               str(seconds)] + enc, check=True)
        return
    rows, t, pcm, sr = [], 0.0, [], 22050
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(sentences):
            w = os.path.join(tmp, f"s{i}.wav")
            dur = synthesize(s, w)
            with wave.open(w, "rb") as fh:
                sr = fh.getframerate()
                pcm.append(fh.readframes(fh.getnframes()))
            rows.append({"start": round(t, 2), "end": round(t + dur, 2), "text": s})
            gap = b"\x00\x00" * int(sr * 0.35)
            pcm.append(gap)
            t += dur + 0.35
        full = os.path.join(tmp, "full.wav")
        with wave.open(full, "wb") as fh:
            fh.setnchannels(1), fh.setsampwidth(2), fh.setframerate(sr)
            fh.writeframes(b"".join(pcm))
        subprocess.run(base + ["-i", full] + enc, check=True)
    with open(os.path.splitext(path)[0] + ".transcript.json", "w") as fh:
        json.dump(rows, fh, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(KB, exist_ok=True)
    for fn, txt in T.KB.items():
        open(os.path.join(KB, fn), "w").write(txt + "\n")

    # --- Scenario 1/2/exact-copy group -------------------------------------
    make_docx("01_market_entry_vietnam.docx",
              [("h", "Market Entry Recommendation: Northwind Foods"), ("p", T.TEXT_A)])
    make_docx("02_vietnam_entry_strategy.docx",   # exact copy, new title/filename
              [("h", "Market Entry Recommendation: Northwind Foods"), ("p", T.TEXT_A)])
    make_docx("12_snack_market_partnering.docx",  # paraphrase of 01
              [("h", "Why Northwind Should Partner Locally"), ("p", T.TEXT_A_PARAPHRASE)])

    # --- cross-modal pair ----------------------------------------------------
    pages = [[("h", p[0])] + [("p", x) for x in p[1:]] for p in T.SC_PAGES]
    make_pdf("03_supply_chain_report.pdf", pages,
             header="Meridian Logistics - Quarterly Operations Review")
    make_video("04_pitch_supply_chain.mp4", T.VIDEO_SC_SENTENCES)

    # --- AI-generated ---------------------------------------------------------
    make_docx("05_ai_education_essay.docx",
              [("h", "AI in Education")] + [("p", x) for x in T.AI_ESSAY])

    # --- shared prompt, original answers (Scenario 4) -------------------------
    def case(answer):
        return [("h", "Section 1: Case Prompt"), ("p", T.CASE_PROMPT),
                ("h", "Section 2: Analysis"), ("p", answer)]
    make_pdf("06_case_alpha.pdf", [case(T.CASE_ANSWERS["alpha"])], header="Case Competition")
    make_docx("07_case_bravo.docx", case(T.CASE_ANSWERS["bravo"]))
    make_pdf("08_case_charlie.pdf", [case(T.CASE_ANSWERS["charlie"])], header="Case Competition")
    make_docx("09_case_delta.docx", case(T.CASE_ANSWERS["delta"]))

    # --- degraded files -----------------------------------------------------
    open(os.path.join(OUT, "10_corrupted_empty.pdf"), "wb").close()      # 0 bytes
    make_video("11_silent_pitch.mp4", [], silent=True)

    # --- originals / external ------------------------------------------------
    make_video("13_original_pitch_bikes.mp4", T.VIDEO_ORIGINAL_SENTENCES)
    kb_par = T.KB["vertical_integration.txt"]
    make_pdf("14_bakery_mill_report.pdf",
             [[("h", "Should Crumb & Co. Buy a Flour Mill?"), ("p", kb_par),
               ("p", T.COMMENTARY_14)]])
    make_pdf("15_urban_beekeeping.pdf", [[("h", "Urban Hive Co."), ("p", T.BEEKEEPING)]])

    manifest = {
        "description": "Ground truth for the synthetic sample set. All text is synthetic.",
        "files": {
            "01_market_entry_vietnam.docx": {"role": "original source A"},
            "02_vietnam_entry_strategy.docx": {"role": "EXACT COPY of 01"},
            "03_supply_chain_report.pdf": {"role": "source of the cross-modal match (page 2)"},
            "04_pitch_supply_chain.mp4": {
                "role": "reads the page-2 paragraph of 03 aloud, one word dropped"},
            "05_ai_education_essay.docx": {"role": "LLM-style essay (AI-generated)"},
            "06_case_alpha.pdf": {"role": "original + shared 150-word prompt"},
            "07_case_bravo.docx": {"role": "original + shared 150-word prompt"},
            "08_case_charlie.pdf": {"role": "original + shared 150-word prompt"},
            "09_case_delta.docx": {"role": "original + shared 150-word prompt"},
            "10_corrupted_empty.pdf": {"role": "degraded: 0-byte PDF"},
            "11_silent_pitch.mp4": {"role": "degraded: MP4 whose audio is silent"},
            "12_snack_market_partnering.docx": {"role": "PARAPHRASE of 01 (synonym rewording)"},
            "13_original_pitch_bikes.mp4": {"role": "original spoken pitch"},
            "14_bakery_mill_report.pdf": {"role": "original commentary + paragraph copied verbatim "
                                                  "from knowledge_base/vertical_integration.txt"},
            "15_urban_beekeeping.pdf": {"role": "original independent report"},
        },
        "peer_overlap_pairs": [
            {"a": "01_market_entry_vietnam.docx", "b": "02_vietnam_entry_strategy.docx",
             "type": "exact_copy"},
            {"a": "01_market_entry_vietnam.docx", "b": "12_snack_market_partnering.docx",
             "type": "paraphrase"},
            {"a": "02_vietnam_entry_strategy.docx", "b": "12_snack_market_partnering.docx",
             "type": "paraphrase"},
            {"a": "03_supply_chain_report.pdf", "b": "04_pitch_supply_chain.mp4",
             "type": "cross_modal"},
        ],
        "external_plagiarism": ["14_bakery_mill_report.pdf"],
        "ai_generated": ["05_ai_education_essay.docx"],
        "boilerplate_group": ["06_case_alpha.pdf", "07_case_bravo.docx",
                              "08_case_charlie.pdf", "09_case_delta.docx"],
        "unprocessable": ["10_corrupted_empty.pdf", "11_silent_pitch.mp4"],
    }
    with open(os.path.join(OUT, "dataset_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("built", len(os.listdir(OUT)), "files in", OUT)


if __name__ == "__main__":
    main()
