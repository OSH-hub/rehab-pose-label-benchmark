#!/usr/bin/env bash
# I7: selector re-justification in the MAIN regime (GT round-0, uniform 30ep).
# Condition = ref (pure GT labels, so the only variable is which frames the
# selector picks). temporal is already covered by the u30 ref run
# (clinician_i4_uniform30); here we run the four non-temporal selectors.
# Each selector gets its own output dir (session tags are strategy-agnostic).
# Runs concurrently with the u30 + review streams. Resume-safe.
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
