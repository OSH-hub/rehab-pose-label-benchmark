"""Evaluate rehabilitation-exercise assessment over skeleton sequences.

Scores predicted vs ground-truth keypoint trajectories with biomechanical and
clinical assessment metrics:

* Per-clinical-joint angle MAE (degrees)
* ROM (range-of-motion) error per joint (degrees)
* Peak-angle error per joint (degrees)
* Temporal smoothness / jerk error (mean jerk difference)
* Proxy deviation score when no assessment labels are available
* Quality-score MAE & Pearson/Spearman correlation (labeled path)
* Correctness F1 (labeled path, binary or multilabel)

All heavy/optional imports (torch, cv2, ultralytics) are guarded inside
functions. Core computation is pure NumPy and always GPU-free.

Usage::

    python -m rehabal_pose.pipelines.evaluate_rehab \\
        --pred outputs/rehab/pred_kps.npy \\
        --gt   outputs/rehab/gt_kps.npy \\
        --out  outputs/rehab/rehab_metrics.json

With optional assessment labels::

    python -m rehabal_pose.pipelines.evaluate_rehab \\
        --pred outputs/rehab/pred_kps.npy \\
        --gt   outputs/rehab/gt_kps.npy \\
        --assessment-labels outputs/rehab/labels.json \\
        --out  outputs/rehab/rehab_metrics.json
"""

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

# ---------------------------------------------------------------------------
# Default rehab26 joint triplets (reused from al/rehabal JOINT_TRIPLETS)
# Indices follow the REHAB24-6 26-keypoint layout confirmed against the real
# Zenodo joints_names.txt (record 13305826).
# ---------------------------------------------------------------------------

#: Clinical joint triplets ``(proximal, vertex, distal)`` for rehab26.
REHAB26_JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow":     (7, 8, 9),       # LeftArm-LeftForeArm-LeftHand
    "right_elbow":    (12, 13, 14),    # RightArm-RightForeArm-RightHand
    "left_knee":      (16, 17, 18),    # LeftUpLeg-LeftLeg-LeftFoot
    "right_knee":     (21, 22, 23),    # RightUpLeg-RightLeg-RightFoot
    "left_shoulder":  (2, 6, 7),       # Spine1-LeftShoulder-LeftArm
    "right_shoulder": (2, 11, 12),     # Spine1-RightShoulder-RightArm
    "left_hip":       (1, 16, 17),     # Spine-LeftUpLeg-LeftLeg
    "right_hip":      (1, 21, 22),     # Spine-RightUpLeg-RightLeg
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_seq(arr: Any) -> np.ndarray:
    """Coerce a keypoint sequence to ``(T, K, D)`` float64.

    Accepts ``(T, K, D)`` directly or ``(K, D)`` (single frame, promoted to
    ``T=1``).
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(
            f"Expected keypoint sequence (T, K, D) or (K, D), got ndim={a.ndim}"
        )
    return a


def _load_npy(path: str | Path) -> np.ndarray:
    """Load a keypoint array from ``.npy`` / ``.npz`` (first array)."""
    p = Path(path)
    obj = np.load(p, allow_pickle=False)
    if hasattr(obj, "files"):  # NpzFile
        if not obj.files:
            raise ValueError(f"No arrays found in {p}")
        return np.asarray(obj[obj.files[0]], dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def _angle_series(kps: np.ndarray, triplet: tuple[int, int, int]) -> np.ndarray:
    """Angle time series at ``triplet[1]`` from ``(T, K, D)`` keypoints (degrees).

    Only the spatial columns (``x, y[, z]``) drive the angle; any trailing
    visibility/confidence channel is dropped via the shared ``_spatial_dim``
    convention so pred-conf vs GT-flag cannot inject a phantom coordinate axis.
    """
    i, j, k = triplet
    s = _spatial_dim(kps.shape[-1])
    return joint_angle(kps[:, i, :s], kps[:, j, :s], kps[:, k, :s])


def _per_joint_rom_and_peak(
    pred_kps: np.ndarray,
    gt_kps: np.ndarray,
    joint_triplets: dict[str, tuple[int, int, int]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-joint ROM error and peak-angle error over a sequence.

    Returns two dicts (``rom_errors``, ``peak_errors``), each mapping joint
    name -> error (degrees). A ``"mean"`` key aggregates NaN-excluded values.
    """
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
    """Absolute difference in temporal smoothness (mean jerk magnitude).

    A smaller value means the predicted trajectory has a similar jerk profile
    to the ground truth. Zero for sequences that are too short (< 4 frames).
    """
    pred_smooth = temporal_smoothness(pred_kps, dt=dt)
    gt_smooth = temporal_smoothness(gt_kps, dt=dt)
    return float(abs(pred_smooth - gt_smooth))


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_rehab(
    pred: Any,
    gt: Any,
    *,
    joint_triplets: Optional[Mapping[str, tuple[int, int, int]]] = None,
    assessment_labels: Optional[dict[str, Any]] = None,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Evaluate rehabilitation-exercise assessment metrics.

    Given predicted and ground-truth keypoint sequences and, optionally,
    clinical assessment labels, this function computes biomechanical and
    assessment metrics for a **single exercise repetition** (i.e. one
    ``(T, K, D)`` sequence pair). To score a whole dataset, call this function
    per sequence and aggregate.

    Parameters
    ----------
    pred, gt:
        Keypoint sequences shaped ``(T, K, D)`` (or ``(K, D)`` for a single
        frame, promoted to ``T=1``). Only the spatial columns ``[:D]`` are
        used; a trailing visibility column is allowed and ignored here.
    joint_triplets:
        Mapping ``name -> (proximal, vertex, distal)`` keypoint-index triplets.
        Defaults to :data:`REHAB26_JOINT_TRIPLETS` (REHAB24-6 26-joint layout).
    assessment_labels:
        Optional dict with any combination of:

        * ``"pred_quality"`` / ``"gt_quality"`` -- 1D arrays of continuous
          quality scores (one per repetition / sample) for MAE & correlation.
        * ``"pred_correctness"`` / ``"gt_correctness"`` -- 1D arrays of binary
          correctness labels (0/1) for F1.

        When ``None`` or absent keys, the proxy path is used and metrics are
        marked ``proxy=True``.
    dt:
        Sampling interval in seconds (used for jerk computation). Default 1.0
        (frame-normalized).

    Returns
    -------
    dict
        Plain, JSON-serialisable metrics dict containing:

        * ``"n_frames"`` -- number of frames ``T``.
        * ``"num_joints"`` -- number of keypoints ``K``.
        * ``"joint_angle_mae"`` -- per-joint angle MAE + ``"mean"`` (degrees).
        * ``"rom_error"`` -- per-joint ROM error + ``"mean"`` (degrees).
        * ``"peak_angle_error"`` -- per-joint peak-angle error + ``"mean"`` (degrees).
        * ``"smoothness_error"`` -- absolute jerk-smoothness difference.
        * ``"assessment"`` -- assessment sub-dict; includes either real metrics
          (``"quality_score_mae"``, ``"quality_score_corr"``,
          ``"correctness_f1"``) or the proxy deviation score with
          ``"proxy": true``.
    """
    triplets = dict(joint_triplets) if joint_triplets is not None else dict(REHAB26_JOINT_TRIPLETS)

    pred_kps = _coerce_seq(pred)
    gt_kps = _coerce_seq(gt)

    if pred_kps.shape[1:] != gt_kps.shape[1:]:
        raise ValueError(
            f"pred/gt joint/dim mismatch: {pred_kps.shape} vs {gt_kps.shape}"
        )

    T = int(pred_kps.shape[0])
    K = int(pred_kps.shape[1])

    # --- per-joint angle MAE --------------------------------------------------
    angle_mae = joint_angle_mae(pred_kps, gt_kps, triplets)

    # --- per-joint ROM error and peak-angle error ----------------------------
    rom_err, peak_err = _per_joint_rom_and_peak(pred_kps, gt_kps, triplets)

    # --- smoothness / jerk error ---------------------------------------------
    smooth_err = _smoothness_error(pred_kps, gt_kps, dt=dt)

    # --- assessment sub-dict -------------------------------------------------
    labels = assessment_labels or {}
    assess: dict[str, Any] = {}

    # Quality score (labeled path)
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
        # Proxy: objective deviation of the predicted sequence from GT.
        assess["proxy_deviation"] = proxy_deviation_score(pred_kps, gt_kps)
        assess["proxy"] = True
        assess["proxy_note"] = (
            "No assessment labels supplied; proxy_deviation is the mean "
            "per-frame per-joint keypoint distance (not a clinical score)."
        )

    # Correctness F1 (labeled path; optional, independent of quality labels)
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


# ---------------------------------------------------------------------------
# File-level helper
# ---------------------------------------------------------------------------


def evaluate_rehab_files(
    pred_path: str | Path,
    gt_path: str | Path,
    out_path: Optional[str | Path] = None,
    *,
    joint_triplets: Optional[Mapping[str, tuple[int, int, int]]] = None,
    assessment_labels_path: Optional[str | Path] = None,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Load ``.npy`` files, evaluate rehab metrics, optionally save JSON.

    Parameters
    ----------
    pred_path, gt_path:
        Paths to predicted and ground-truth keypoint ``.npy`` / ``.npz``
        arrays shaped ``(T, K, D)``.
    out_path:
        Optional path for the output metrics JSON.
    joint_triplets:
        Forwarded to :func:`evaluate_rehab`.
    assessment_labels_path:
        Optional path to a JSON file whose keys match the ``assessment_labels``
        parameter of :func:`evaluate_rehab` (``pred_quality``, ``gt_quality``,
        ``pred_correctness``, ``gt_correctness``).
    dt:
        Sampling interval; forwarded to :func:`evaluate_rehab`.
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: score predicted vs GT keypoint sequences, write JSON."""
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
