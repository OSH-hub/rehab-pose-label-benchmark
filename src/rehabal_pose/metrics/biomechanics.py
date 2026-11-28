from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "JOINT_TRIPLETS",
    "joint_angle",
    "compute_rom",
    "joint_angle_mae",
    "rom_error",
    "peak_angle_error",
    "phase_error",
    "jerk",
    "temporal_smoothness",
]

_EPS = 1e-12

JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "right_elbow": (12, 11, 10),
    "left_elbow": (13, 14, 15),
    "right_shoulder": (11, 12, 2),
    "left_shoulder": (14, 13, 3),
    "right_knee": (2, 1, 0),
    "left_knee": (3, 4, 5),
    "right_hip": (1, 2, 7),
    "left_hip": (4, 3, 7),
}


def joint_angle(a: Any, b: Any, c: Any, *, degrees: bool = True) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba, axis=-1)
    nbc = np.linalg.norm(bc, axis=-1)
    denom = nba * nbc
    dot = np.sum(ba * bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(denom > _EPS, dot / np.where(denom > _EPS, denom, 1.0), np.nan)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos)
    if degrees:
        ang = np.degrees(ang)
    return ang


def compute_rom(angle_series: Any) -> dict[str, Any]:
    arr = np.asarray(angle_series, dtype=np.float64).ravel()
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {"min": float("nan"), "max": float("nan"), "rom": float("nan"), "peak_idx": -1}
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    peak_idx = int(np.nanargmax(arr))
    return {"min": amin, "max": amax, "rom": amax - amin, "peak_idx": peak_idx}


def _spatial_dim(c: int) -> int:
    if c <= 2:
        return c
    if c == 3:
        return 2
    if c == 4:
        return 3
    return c - 1


def _angle_series_from_kps(
    kps: np.ndarray, triplet: tuple[int, int, int]
) -> np.ndarray:
    i, j, k = triplet
    s = _spatial_dim(kps.shape[-1])
    return joint_angle(kps[:, i, :s], kps[:, j, :s], kps[:, k, :s])


def joint_angle_mae(
    pred_kps: Any,
    gt_kps: Any,
    joint_triplets: Optional[Mapping[str, tuple[int, int, int]]] = None,
) -> dict[str, float]:
    triplets = dict(joint_triplets) if joint_triplets is not None else dict(JOINT_TRIPLETS)
    pred = np.asarray(pred_kps, dtype=np.float64)
    gt = np.asarray(gt_kps, dtype=np.float64)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.ndim == 2:
        gt = gt[None, ...]
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    out: dict[str, float] = {}
    per_values: list[float] = []
    for name, trip in triplets.items():
        pa = _angle_series_from_kps(pred, trip)
        ga = _angle_series_from_kps(gt, trip)
        diff = np.abs(pa - ga)
        if np.all(np.isnan(diff)):
            mae = float("nan")
        else:
            mae = float(np.nanmean(diff))
            per_values.append(mae)
        out[name] = mae
    out["mean"] = float(np.mean(per_values)) if per_values else float("nan")
    return out


def rom_error(pred_angles: Any, gt_angles: Any) -> float:
    p = compute_rom(pred_angles)["rom"]
    g = compute_rom(gt_angles)["rom"]
    return float(abs(p - g))


def peak_angle_error(pred_angles: Any, gt_angles: Any) -> float:
    p = compute_rom(pred_angles)["max"]
    g = compute_rom(gt_angles)["max"]
    return float(abs(p - g))


def phase_error(
    pred_angles: Any, gt_angles: Any, *, normalize: bool = False
) -> float:
    pi = compute_rom(pred_angles)["peak_idx"]
    gi = compute_rom(gt_angles)["peak_idx"]
    if pi < 0 or gi < 0:
        return float("nan")
    err = float(abs(pi - gi))
    if normalize:
        length = max(np.asarray(gt_angles).ravel().size, 1)
        err /= length
    return err


def jerk(traj: Any, *, dt: float = 1.0) -> np.ndarray:
    a = np.asarray(traj, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < 4:
        return np.empty((0,) + a.shape[1:-1], dtype=np.float64)
    d3 = np.diff(a, n=3, axis=0) / (dt ** 3)
    return np.linalg.norm(d3, axis=-1)


def temporal_smoothness(traj: Any, *, dt: float = 1.0) -> float:
    j = jerk(traj, dt=dt)
    if j.size == 0:
        return 0.0
    return float(-np.mean(np.abs(j)))
