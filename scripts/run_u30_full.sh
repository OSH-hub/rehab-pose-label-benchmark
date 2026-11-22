#!/usr/bin/env bash
# Full uniform-30ep run (I4 schedule applied to the whole experiment).
# Sequential so each stage owns both GPUs; every stage is resume-safe, so
# re-running this script after an interruption continues where it stopped.
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
LOG=outputs/experiments/u30_full.log
R=scripts/run_clinician_isolated.py
B=scripts/run_clinician_experiment.py
E="--epochs 30 --epochs-round0 30"

echo "=== STEP1 11n sigma sweep (B sig5/10/20/40) $(date) ===" >> "$LOG"
python $R --seeds 0 1 2 3 4 5 --conditions B --sigmas 5 10 20 40 --gpus 0 1 $E \
  --out outputs/experiments/clinician_i4_uniform30 >> "$LOG" 2>&1

echo "=== STEP2 11n full-grid re-aggregation $(date) ===" >> "$LOG"
python $B --seeds 0 1 2 3 4 5 --conditions A B ref --sigmas 0 5 10 20 40 $E \
  --out outputs/experiments/clinician_i4_uniform30 >> "$LOG" 2>&1

echo "=== STEP3 11s full (A/B sig0/ref) $(date) ===" >> "$LOG"
python $R --seeds 0 1 2 3 4 5 --gpus 0 1 --weights yolo11s-pose.pt $E \
  --out outputs/experiments/clinician_s_u30 >> "$LOG" 2>&1

echo "=== STEP4 11m full (A/B sig0/ref) $(date) ===" >> "$LOG"
python $R --seeds 0 1 2 3 4 5 --gpus 0 1 --weights yolo11m-pose.pt $E \
  --out outputs/experiments/clinician_m_u30 >> "$LOG" 2>&1

echo "ALL_U30_DONE $(date)" >> "$LOG"
