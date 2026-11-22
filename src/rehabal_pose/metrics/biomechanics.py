"""Biomechanical / kinematic metrics for rehabilitation assessment (NumPy).

These metrics turn raw keypoint trajectories into clinically meaningful joint
angles and range-of-motion (ROM) statistics, and quantify how well a predicted
motion matches a reference one:

* :func:`joint_angle` -- angle (deg) at the middle of a 3-point chain, 2D/3D,
  vectorized over arbitrary leading (time/batch) dimensions.
* :func:`compute_rom` -- min / max / range / peak index of an angle series.
* :func:`joint_angle_mae` -- per-joint mean absolute angular error.
* :func:`rom_error`, :func:`peak_angle_error`, :func:`phase_error`.
* :func:`jerk` and :func:`temporal_smoothness` -- motion-quality measures.

A default :data:`JOINT_TRIPLETS` mapping (MPII-style 16-keypoint ordering) is
provided for the common rehab joints; every function that consumes triplets
accepts a custom mapping so it can be re-parameterized per dataset/skeleton.
"""

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

#: Default joint -> (proximal, vertex, distal) keypoint-index triplets.
#:
#: Indices follow the canonical MPII 16-keypoint ordering, which the AIHub 219
#: 16-keypoint skeleton mirrors::
#:
#:   0  r_ankle     4  l_knee     8  upper_neck   12 r_shoulder
#:   1  r_knee      5  l_ankle    9  head_top     13 l_shoulder
#:   2  r_hip       6  pelvis    10  r_wrist      14 l_elbow
#:   3  l_hip       7  thorax    11  r_elbow      15 l_wrist
#:
#: The angle is measured at the *vertex* (middle) joint. Pass a custom mapping
#: to any consumer to adapt to a different skeleton.
JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    "right_elbow": (12, 11, 10),  # shoulder - elbow - wrist
    "left_elbow": (13, 14, 15),
    "right_shoulder": (11, 12, 2),  # elbow - shoulder - hip
    "left_shoulder": (14, 13, 3),
    "right_knee": (2, 1, 0),  # hip - knee - ankle
    "left_knee": (3, 4, 5),
    "right_hip": (1, 2, 7),  # knee - hip - thorax
    "left_hip": (4, 3, 7),
}


def joint_angle(a: Any, b: Any, c: Any, *, degrees: bool = True) -> np.ndarray:
    """Angle at ``b`` formed by the chain ``a - b - c``.

    Works for 2D or 3D points and is vectorized over any leading dimensions:
    pass ``a, b, c`` shaped ``(..., D)`` (e.g. ``(T, D)`` for a time series) and
    the result is shaped ``(...)``.

    Parameters
    ----------
    a, b, c:
        Point coordinates with matching shapes ``(..., D)``, ``D in (2, 3)``.
        ``b`` is the vertex at which the angle is measured.
    degrees:
        Return degrees (default) or radians.

    Returns
    -------
    numpy.ndarray
        Angle(s) shaped like the leading dims. Degenerate (zero-length) limbs
        yield ``nan``.
    """
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
    """Range-of-motion statistics for a 1D angle time series.

    Parameters
    ----------
    angle_series:
        Sequence of joint angles over time (NaNs are ignored).

    Returns
    -------
    dict
        ``{"min": float, "max": float, "rom": float, "peak_idx": int}`` where
        ``rom = max - min`` and ``peak_idx`` is the index of the maximum. An
        all-NaN/empty series yields NaN stats and ``peak_idx = -1``.
    """
    arr = np.asarray(angle_series, dtype=np.float64).ravel()
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {"min": float("nan"), "max": float("nan"), "rom": float("nan"), "peak_idx": -1}
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    peak_idx = int(np.nanargmax(arr))
    return {"min": amin, "max": amax, "rom": amax - amin, "peak_idx": peak_idx}


def _spatial_dim(c: int) -> int:
    """Number of leading *spatial* columns for a ``C``-channel keypoint array.

    Mirrors the repo convention (see :func:`rehabal_pose.metrics.pose._split_coords_vis`):
    ``C == 2`` -> 2D, ``C == 3`` -> 2D + visibility, ``C == 4`` -> 3D + visibility.
    The trailing visibility/confidence channel must NOT enter angle math: pred
    visibility (YOLO conf ~0.9) and GT visibility (label flag in {0, 1, 2}) live on
    different scales, so feeding it as a coordinate injects a phantom axis that
    systematically corrupts the joint angle.
    """
    if c <= 2:
        return c
    if c == 3:
        return 2
    if c == 4:
        return 3
    return c - 1  # C > 4: treat the final channel as visibility, the rest as spatial


def _angle_series_from_kps(
    kps: np.ndarray, triplet: tuple[int, int, int]
) -> np.ndarray:
    """Compute an angle time series for one triplet from ``(T, K, D)`` kps.

    Only the spatial columns (``x, y[, z]``) drive the angle; any trailing
    visibility/confidence channel is dropped (see :func:`_spatial_dim`).
    """
    i, j, k = triplet
    s = _spatial_dim(kps.shape[-1])
    return joint_angle(kps[:, i, :s], kps[:, j, :s], kps[:, k, :s])


def joint_angle_mae(
    pred_kps: Any,
    gt_kps: Any,
    joint_triplets: Optional[Mapping[str, tuple[int, int, int]]] = None,
) -> dict[str, float]:
    """Per-joint mean absolute angular error (degrees).

    For each named triplet an angle time series is derived from both predicted
    and ground-truth keypoints and the mean absolute difference is returned.

    Parameters
    ----------
    pred_kps, gt_kps:
        Keypoint trajectories shaped ``(T, K, D)`` (a single frame ``(K, D)``
        is promoted to ``T = 1``). Only the spatial columns are used.
    joint_triplets:
        Mapping ``name -> (i, j, k)``. Defaults to :data:`JOINT_TRIPLETS`.

    Returns
    -------
    dict
        ``name -> MAE`` plus a ``"mean"`` key averaging the per-joint MAEs
        (NaN-joints excluded from the mean).
    """
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
    """Absolute error between predicted and GT range of motion (degrees)."""
    p = compute_rom(pred_angles)["rom"]
    g = compute_rom(gt_angles)["rom"]
    return float(abs(p - g))


def peak_angle_error(pred_angles: Any, gt_angles: Any) -> float:
    """Absolute error between predicted and GT peak (maximum) angle (degrees)."""
    p = compute_rom(pred_angles)["max"]
    g = compute_rom(gt_angles)["max"]
    return float(abs(p - g))


def phase_error(
    pred_angles: Any, gt_angles: Any, *, normalize: bool = False
) -> float:
    """Timing error between the peak of two angle series.

    Parameters
    ----------
    pred_angles, gt_angles:
        Angle time series.
    normalize:
        If ``True`` divide the index difference by the GT series length to
        return a fraction-of-cycle phase error in ``[0, 1]``.

    Returns
    -------
    float
        ``|peak_idx_pred - peak_idx_gt|`` in frames, optionally normalized.
    """
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
    """Jerk magnitude (3rd time-derivative) of a trajectory.

    Parameters
    ----------
    traj:
        Trajectory shaped ``(T, ...)`` -- e.g. ``(T, D)`` for a single point or
        ``(T, K, D)`` for a multi-keypoint sequence. Differentiation is along
        axis 0 (time).
    dt:
        Sampling interval (seconds). Jerk scales as ``1/dt^3``.

    Returns
    -------
    numpy.ndarray
        Per-time-step jerk magnitude shaped ``(T-3, ...)`` collapsed over the
        trailing coordinate axis (Euclidean norm). Empty if ``T < 4``.
    """
    a = np.asarray(traj, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < 4:
        return np.empty((0,) + a.shape[1:-1], dtype=np.float64)
    d3 = np.diff(a, n=3, axis=0) / (dt ** 3)
    return np.linalg.norm(d3, axis=-1)


def temporal_smoothness(traj: Any, *, dt: float = 1.0) -> float:
    """Scalar motion-smoothness score (mean jerk magnitude, negated).

    Higher (closer to 0) is smoother. Defined as ``-mean(|jerk|)`` so that
    smoother trajectories score higher, consistent with "smoothness" semantics.
    Returns ``0.0`` for trajectories too short to define jerk.

    Parameters
    ----------
    traj:
        Trajectory ``(T, ...)`` as accepted by :func:`jerk`.
    dt:
        Sampling interval.
    """
    j = jerk(traj, dt=dt)
    if j.size == 0:
        return 0.0
    return float(-np.mean(np.abs(j)))
