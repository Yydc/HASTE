#!/usr/bin/env bash
# Evaluate a trained HASTE checkpoint
# Usage: bash scripts/evaluate.sh --config configs/seedvig.yaml --checkpoint checkpoints/hast_seedvig_fold0_seed2025.pth [--fold 0]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python "$ROOT_DIR/train.py" \
    --eval-only \
    "$@"
