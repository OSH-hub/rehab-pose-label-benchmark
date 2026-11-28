#!/usr/bin/env bash
set -u
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONIOENCODING=utf-8

ROOT="${CLIN_ROOT:-outputs/experiments/clinician}"
VAL="outputs/rehab24_6_yolo/images/val"
LAB="outputs/rehab24_6_yolo/labels/val"
OUTDIR="${DS_OUTDIR:-outputs/experiments/downstream}"
BUDGETS="0.05 0.1 0.2 0.4"
mkdir -p "$OUTDIR"

declare -A DIRSUFFIX=( [A]="A_s%s" [B]="B_s%s_sig0" [ref]="ref_s%s" )

SEEDS=("$@")
echo "[downstream] GPU=${CUDA_VISIBLE_DEVICES:-?} seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  for cond in A B ref; do
    dir=$(printf "${DIRSUFFIX[$cond]}" "$seed")
    sess="$ROOT/$dir"
    out="$OUTDIR/${cond}_s${seed}_downstream.json"
    if [[ -f "$out" ]]; then
      echo "[skip] $out exists"; continue
    fi
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
