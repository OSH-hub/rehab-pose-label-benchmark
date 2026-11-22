#!/usr/bin/env bash
# Supplemental ablation experiments (I2/I1/I3), uniform 30ep, run concurrently with
# the u30_full run to use the spare GPU/CPU headroom. Each stage is resume-safe.
# Anchors (A/B/ref at uniform 30ep) come from clinician_i4_uniform30 (u30 run).
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
LOG=outputs/experiments/review_experiments.log
R=scripts/run_clinician_isolated.py
E="--epochs 30 --epochs-round0 30"

echo "=== I2 filter x correction decomposition $(date) ===" >> "$LOG"
python $R --seeds 0 1 2 3 4 5 \
  --conditions A_filter ref_filter B_reject_only B_correct_only \
  --gpus 0 1 $E --out outputs/experiments/clinician_i2_decomp >> "$LOG" 2>&1

echo "=== I1 selective-correction B (keep OKS >= 0.5) $(date) ===" >> "$LOG"
python $R --seeds 0 1 2 3 4 5 --conditions B --verify-oks-threshold 0.5 \
  --gpus 0 1 $E --out outputs/experiments/clinician_i1_verify >> "$LOG" 2>&1

echo "=== I3 confidence-gated pseudo-label A_conf (score >= 0.5) $(date) ===" >> "$LOG"
python $R --seeds 0 1 2 3 4 5 --conditions A_conf --conf-threshold 0.5 \
  --gpus 0 1 $E --out outputs/experiments/clinician_i3_conf >> "$LOG" 2>&1

echo "ALL_REVIEW_DONE $(date)" >> "$LOG"
