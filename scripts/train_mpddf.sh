#!/usr/bin/env bash
# Train HASTE on MPD-DF dataset
# Usage: bash scripts/train_mpddf.sh [--fold 0] [extra args...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python "$ROOT_DIR/train.py" \
    --config "$ROOT_DIR/configs/mpddf.yaml" \
    "$@"
