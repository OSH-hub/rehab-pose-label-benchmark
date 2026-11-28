from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..metrics.biomechanics import (
    _spatial_dim,
    joint_angle,
    joint_angle_mae,
    rom_error,
    peak_angle_error,
    temporal_smoothness,
)
from ..metrics.assessment import (
    proxy_deviation_score,
    quality_score_mae,
    quality_score_corr,
    compensation_f1,
)
from ..utils.io import write_json
from ..utils.logging import get_logger

__all__ = [
    "evaluate_rehab",
    "evaluate_rehab_files",
    "main",
]

_LOG = get_logger(__name__)


REHAB26_JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow":     (7, 8, 9),
    "right_elbow":    (12, 13, 14),
    "left_knee":      (16, 17, 18),
    "right_knee":     (21, 22, 23),
    "left_shoulder":  (2, 6, 7),
    "right_shoulder": (2, 11, 12),
    "left_hip":       (1, 16, 17),
    "right_hip":      (1, 21, 22),
}


def _coerce_seq(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(
            f"Expected keypoint sequence (T, K, D) or (K, D), got ndim={a.ndim}"
        )
    return a


def _load_npy(path: str | Path) -> np.ndarray:
    p = Path(path)
    obj = np.load(p, allow_pickle=False)
    if hasattr(obj, "files"):
        if not obj.files:
            raise ValueError(f"No arrays found in {p}")
        return np.asarray(obj[obj.files[0]], dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def _angle_series(kps: np.ndarray, triplet: tuple[int, int, int]) -> np.ndarray:
    i, j, k = triplet
    s = _spatial_dim(kps.shape[-1])
    return joint_angle(kps[:, i, :s], kps[:, j, :s], kps[:, k, :s])


def _per_joint_rom_and_peak(
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    joint_triplets: dict[str, tuple[int, int, int]],
) -> tuple[dict[str, float], dict[str, float]]:
    rom_errors: dict[str, float] = {}
    peak_errors: dict[str, float] = {}
    rom_vals: list[float] = []
    peak_vals: list[float] = []

    for name, trip in joint_triplets.items():
        pred_angles = _angle_series(pred_kps, trip)
        gt_angles = _angle_series(gt_kps, trip)
        re = rom_error(pred_angles, gt_angles)
        pe = peak_angle_error(pred_angles, gt_angles)
        rom_errors[name] = re
        peak_errors[name] = pe
        if not np.isnan(re):
            rom_vals.append(re)
        if not np.isnan(pe):
            peak_vals.append(pe)

    rom_errors["mean"] = float(np.mean(rom_vals)) if rom_vals else float("nan")
    peak_errors["mean"] = float(np.mean(peak_vals)) if peak_vals else float("nan")
    return rom_errors, peak_errors


def _smoothness_error(
    pred_kps: np.ndarray, gt_kps: np.ndarray, *, dt: float = 1.0
) -> float:
    pred_smooth = temporal_smoothness(pred_kps, dt=dt)
    gt_smooth = temporal_smoothness(gt_kps, dt=dt)
    return float(abs(pred_smooth - gt_smooth))


def evaluate_rehab(
    pred: Any,
    gt: Any,
    *,
    joint_triplets: Optional[Mapping[str, tuple[int, int, int]]] = None,
    assessment_labels: Optional[dict[str, Any]] = None,
    dt: float = 1.0,
) -> dict[str, Any]:
    triplets = dict(joint_triplets) if joint_triplets is not None else dict(REHAB26_JOINT_TRIPLETS)

    pred_kps = _coerce_seq(pred)
    gt_kps = _coerce_seq(gt)

    if pred_kps.shape[1:] != gt_kps.shape[1:]:
        raise ValueError(
            f"pred/gt joint/dim mismatch: {pred_kps.shape} vs {gt_kps.shape}"
        )

    T = int(pred_kps.shape[0])
    K = int(pred_kps.shape[1])

    angle_mae = joint_angle_mae(pred_kps, gt_kps, triplets)

    rom_err, peak_err = _per_joint_rom_and_peak(pred_kps, gt_kps, triplets)

    smooth_err = _smoothness_error(pred_kps, gt_kps, dt=dt)

    labels = assessment_labels or {}
    assess: dict[str, Any] = {}

    if "pred_quality" in labels and "gt_quality" in labels:
        pred_q = np.asarray(labels["pred_quality"], dtype=np.float64).ravel()
        gt_q = np.asarray(labels["gt_quality"], dtype=np.float64).ravel()
        assess["quality_score_mae"] = quality_score_mae(pred_q, gt_q)
        assess["quality_score_corr_pearson"] = quality_score_corr(
            pred_q, gt_q, method="pearson"
        )
        assess["quality_score_corr_spearman"] = quality_score_corr(
            pred_q, gt_q, method="spearman"
        )
        assess["proxy"] = False
    else:
        assess["proxy_deviation"] = proxy_deviation_score(pred_kps, gt_kps)
        assess["proxy"] = True
        assess["proxy_note"] = (
            "No assessment labels supplied; proxy_deviation is the mean "
            "per-frame per-joint keypoint distance (not a clinical score)."
        )

    if "pred_correctness" in labels and "gt_correctness" in labels:
        f1_res = compensation_f1(
            labels["pred_correctness"], labels["gt_correctness"]
        )
        assess["correctness_f1"] = f1_res["f1"]
        assess["correctness_precision"] = f1_res["precision"]
        assess["correctness_recall"] = f1_res["recall"]

    return {
        "n_frames": T,
        "num_joints": K,
        "joint_angle_mae": angle_mae,
        "rom_error": rom_err,
        "peak_angle_error": peak_err,
        "smoothness_error": smooth_err,
        "assessment": assess,
    }


def evaluate_rehab_files(
    pred_path: str | Path,
    gt_path: str | Path,
    out_path: Optional[str | Path] = None,
    *,
    joint_triplets: Optional[Mapping[str, tuple[int, int, int]]] = None,
    assessment_labels_path: Optional[str | Path] = None,
    dt: float = 1.0,
) -> dict[str, Any]:
    pred = _load_npy(pred_path)
    gt = _load_npy(gt_path)

    labels: Optional[dict[str, Any]] = None
    if assessment_labels_path is not None:
        p = Path(assessment_labels_path)
        with p.open("r", encoding="utf-8") as fh:
            labels = json.load(fh)

    metrics = evaluate_rehab(
        pred, gt,
        joint_triplets=joint_triplets,
        assessment_labels=labels,
        dt=dt,
    )

    if out_path is not None:
        write_json(out_path, metrics)
        _LOG.info("Wrote rehab metrics -> %s", out_path)

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rehabilitation-exercise assessment metrics "
            "(joint-angle MAE, ROM error, peak-angle error, smoothness, "
            "proxy deviation / quality-score MAE & F1)."
        )
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="predicted keypoint sequence .npy (T,K,D)",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="ground-truth keypoint sequence .npy (T,K,D)",
    )
    parser.add_argument(
        "--assessment-labels",
        default=None,
        metavar="JSON",
        help=(
            "optional JSON with pred_quality / gt_quality / "
            "pred_correctness / gt_correctness arrays"
        ),
    )
    parser.add_argument(
        "--skeleton",
        default="rehab26",
        help=(
            "skeleton name for default joint triplets "
            "(only 'rehab26' recognized; ignored when --joint-triplets is given)"
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="sampling interval in seconds for jerk computation (default 1.0)",
    )
    parser.add_argument("--out", default=None, help="output metrics .json")
    args = parser.parse_args(argv)

    metrics = evaluate_rehab_files(
        args.pred,
        args.gt,
        args.out,
        assessment_labels_path=args.assessment_labels,
        dt=args.dt,
    )

    if args.out is None:
        angle_mean = metrics["joint_angle_mae"].get("mean", float("nan"))
        rom_mean = metrics["rom_error"].get("mean", float("nan"))
        _LOG.info(
            "angle_MAE_mean=%.3f  ROM_err_mean=%.3f  smoothness_err=%.6f",
            angle_mean,
            rom_mean,
            metrics["smoothness_error"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
