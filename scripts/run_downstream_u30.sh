#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
LOG=outputs/experiments/downstream_u30.log

run_track() {  # $1 clin_root  $2 ds_outdir
  echo "=== downstream $2 $(date) ===" >> "$LOG"
  CLIN_ROOT="$1" DS_OUTDIR="$2" CUDA_VISIBLE_DEVICES=0 bash scripts/run_downstream_all.sh 0 1 2 >> "$LOG" 2>&1 &
  CLIN_ROOT="$1" DS_OUTDIR="$2" CUDA_VISIBLE_DEVICES=1 bash scripts/run_downstream_all.sh 3 4 5 >> "$LOG" 2>&1 &
  wait
}

run_track outputs/experiments/clinician_i4_uniform30 outputs/experiments/downstream_11n_u30
run_track outputs/experiments/clinician_s_u30       outputs/experiments/downstream_s_u30
run_track outputs/experiments/clinician_m_u30       outputs/experiments/downstream_m_u30
echo "ALL_DOWNSTREAM_U30_DONE $(date)" >> "$LOG"
