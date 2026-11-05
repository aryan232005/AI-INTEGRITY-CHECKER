#!/usr/bin/env python3
"""Find out why Whisper fails.   python scripts/debug_whisper.py [video.mp4]

Runs each step separately and prints the full error for the first one that breaks.
"""
import importlib.metadata as md
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

video = sys.argv[1] if len(sys.argv) > 1 else "sample_submissions/04_pitch_supply_chain.mp4"
print("Package versions:")
PACKAGES = ("faster-whisper", "ctranslate2", "huggingface_hub", "tokenizers",
            "onnxruntime", "av", "numpy")
for pkg in PACKAGES:
    try:
        print(f"  {pkg:16} {md.version(pkg)}")
    except Exception:
        print(f"  {pkg:16} NOT INSTALLED")

from src.extractor import _load_wav_float32, extract_audio  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    wav = os.path.join(tmp, "a.wav")
    print("\nStep 1: extract audio with ffmpeg ...")
    try:
        extract_audio(video, wav)
        print("  OK")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("Step 2: load Whisper model ...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("  OK")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("Step 3: transcribe (with vad_filter) ...")
    try:
        segs, _ = model.transcribe(_load_wav_float32(wav), language="en", vad_filter=True)
        for s in segs:
            print(f"  [{s.start:5.1f}-{s.end:5.1f}] {s.text.strip()}")
        print("  OK")
    except Exception:
        traceback.print_exc()
        print("\nStep 3b: retry WITHOUT vad_filter ...")
        try:
            segs, _ = model.transcribe(_load_wav_float32(wav), language="en")
            for s in segs:
                print(f"  [{s.start:5.1f}-{s.end:5.1f}] {s.text.strip()}")
            print("  OK without VAD -> the VAD step is the problem")
        except Exception:
            traceback.print_exc()
