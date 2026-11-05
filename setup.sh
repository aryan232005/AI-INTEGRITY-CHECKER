#!/usr/bin/env bash
# One-command setup + demo run:   bash setup.sh
set -euo pipefail
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (apt install ffmpeg | brew install ffmpeg)"; exit 1; }
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
python scripts/download_models.py
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --knowledge_base ./knowledge_base
python scripts/evaluate.py
