#!/usr/bin/env bash
# Drive per-(condition,seed) downstream rehab-metric eval across one GPU.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_downstream_all.sh 0 1 2
#        CUDA_VISIBLE_DEVICES=1 bash scripts/run_downstream_all.sh 3 4
# Override the experiment grid for a different backbone run via env:
#   CLIN_ROOT=outputs/experiments/clinician_s \
#   DS_OUTDIR=outputs/experiments/downstream_s \
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_downstream_all.sh 0 1 2
# Resume-safe: skips a (cond,seed) whose output JSON already exists.
set -u
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8

ROOT="${CLIN_ROOT:-outputs/experiments/clinician}"
VAL="outputs/rehab24_6_yolo/images/val"
LAB="outputs/rehab24_6_yolo/labels/val"
OUTDIR="${DS_OUTDIR:-outputs/experiments/downstream}"
BUDGETS="0.05 0.1 0.2 0.4"
mkdir -p "$OUTDIR"

# condition tag -> session-dir suffix
declare -A DIRSUFFIX=( [A]="A_s%s" [B]="B_s%s_sig0" [ref]="ref_s%s" )

SEEDS=("$@")
echo "[downstream] GPU=${CUDA_VISIBLE_DEVICES:-?} seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  for cond in A B ref; do
    # shellcheck disable=SC2059
    dir=$(printf "${DIRSUFFIX[$cond]}" "$seed")
    sess="$ROOT/$dir"
    out="$OUTDIR/${cond}_s${seed}_downstream.json"
    if [[ -f "$out" ]]; then
      echo "[skip] $out exists"; continue
    fi
    # collect the 4 per-budget best.pt (round_0..3)
    wlist=()
    for r in 0 1 2 3; do
      w="$sess/round_${r}/run/train/weights/best.pt"
      [[ -f "$w" ]] && wlist+=("$w")
    done
    if [[ ${#wlist[@]} -ne 4 ]]; then
      echo "[warn] $sess: found ${#wlist[@]}/4 best.pt — skipping"; continue
    fi
    echo "[run ] GPU=${CUDA_VISIBLE_DEVICES:-?} $cond seed=$seed -> $out"
    python scripts/run_downstream_eval.py \
      --weights-list "${wlist[@]}" \
      --budgets $BUDGETS \
      --val-dir "$VAL" --labels-dir "$LAB" \
      --out "$out" \
      && echo "[ok  ] $out" \
      || echo "[FAIL] $cond seed=$seed (continuing)"
  done
done
echo "[downstream] GPU=${CUDA_VISIBLE_DEVICES:-?} DONE seeds=${SEEDS[*]}"
