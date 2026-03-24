#!/usr/bin/env bash
# Run LOSO training across all folds for a given dataset config.
# Usage: bash scripts/run_all_folds.sh configs/seedvig.yaml [extra args...]
#
# Example:
#   bash scripts/run_all_folds.sh configs/seedvig.yaml --epochs 50
#   bash scripts/run_all_folds.sh configs/sadt.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [extra args...]"
    exit 1
fi

CONFIG="$1"
shift

# Extract n_folds from the YAML config
N_FOLDS=$(python -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('n_folds', 5))
")

echo "============================================"
echo "HASTE LOSO Training: $CONFIG ($N_FOLDS folds)"
echo "============================================"

RESULTS_FILE="results_$(basename "$CONFIG" .yaml).txt"
> "$RESULTS_FILE"

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo ""
    echo "--- Fold $FOLD / $N_FOLDS ---"
    python "$ROOT_DIR/train.py" \
        --config "$CONFIG" \
        --fold "$FOLD" \
        "$@" 2>&1 | tee -a "$RESULTS_FILE"
    echo "--- Fold $FOLD complete ---"
done

echo ""
echo "============================================"
echo "All $N_FOLDS folds complete."
echo "Results saved to: $RESULTS_FILE"
echo "============================================"
