#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
LOG=outputs/experiments/i7_selectors.log
R=scripts/run_clinician_isolated.py
E="--epochs 30 --epochs-round0 30"

for S in random uncertainty coreset rehabal; do
  echo "=== I7 selector=$S (ref, uniform 30ep) $(date) ===" >> "$LOG"
  python $R --seeds 0 1 2 3 4 5 --conditions ref --strategy "$S" \
    --gpus 0 1 $E --out "outputs/experiments/clinician_i7_$S" >> "$LOG" 2>&1
done
echo "ALL_I7_DONE $(date)" >> "$LOG"
