#!/usr/bin/env python3
"""Pre-download all model weights so the pipeline can run offline afterwards.

    python scripts/download_models.py                 # MiniLM + Whisper-base
    python scripts/download_models.py --whisper tiny  # smaller ASR model
    python scripts/download_models.py --with-roberta  # optional AI-text detector (~500 MB)
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="base", choices=["tiny", "base", "small"])
    ap.add_argument("--with-roberta", action="store_true")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    print("Downloading sentence-transformers/all-MiniLM-L6-v2 ...")
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    from faster_whisper import WhisperModel
    print(f"Downloading Whisper '{a.whisper}' (faster-whisper / CTranslate2) ...")
    WhisperModel(a.whisper, device="cpu", compute_type="int8")

    if a.with_roberta:
        from transformers import pipeline
        print("Downloading openai-community/roberta-base-openai-detector ...")
        pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
    print("Done.")


if __name__ == "__main__":
    main()
