#!/usr/bin/env bash
# Run LOSO training across all folds and 3 seeds (paper Table 2) for a given dataset config.
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
SEEDS="${SEEDS:-2025 2026 2027}"

# Extract n_folds from the YAML config
N_FOLDS=$(python -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('n_folds', 5))
")

echo "============================================"
echo "HASTE LOSO Training: $CONFIG ($N_FOLDS folds, seeds: $SEEDS)"
echo "============================================"

RESULTS_FILE="results_$(basename "$CONFIG" .yaml).txt"
> "$RESULTS_FILE"

for SEED in $SEEDS; do
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    echo ""
    echo "--- Seed $SEED | Fold $FOLD / $N_FOLDS ---"
    python "$ROOT_DIR/train.py" \
        --config "$CONFIG" \
        --fold "$FOLD" \
        --seed "$SEED" \
        "$@" 2>&1 | tee -a "$RESULTS_FILE"
    echo "--- Seed $SEED | Fold $FOLD complete ---"
done
done

echo ""
echo "============================================"
echo "All $N_FOLDS folds complete."
echo "Results saved to: $RESULTS_FILE"
echo "============================================"
