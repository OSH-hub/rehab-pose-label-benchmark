"""Pose-estimation accuracy metrics (pure NumPy).

Implements the standard keypoint-localization metrics used to evaluate the
2D/3D pose detectors in RehabAL-Pose:

* :func:`pck` -- Percentage of Correct Keypoints (per-joint + mean).
* :func:`oks` -- Object Keypoint Similarity (COCO-style, per sample).
* :func:`mean_per_joint_error` -- MPJPE / mean Euclidean localization error.

All functions operate on arrays shaped ``(N, K, C)`` where ``N`` is the number
of samples (or frames), ``K`` the number of keypoints, and ``C`` the coordinate
dimension (``2`` for 2D, ``3`` for 3D; an extra visibility channel is allowed
and only the leading ``C-1`` / spatial columns are used for distances when a
visibility channel is detected). No deep-learning dependencies are imported.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = [
    "pck",
    "oks",
    "mean_per_joint_error",
]

_EPS = 1e-12


def _as_3d(arr: Any) -> np.ndarray:
    """Coerce keypoint input to a ``(N, K, C)`` float64 array.

    A single sample of shape ``(K, C)`` is promoted to ``(1, K, C)``.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(
            f"Expected keypoints of shape (N, K, C) or (K, C), got ndim={a.ndim}"
        )
    return a


def _split_coords_vis(
    pred: np.ndarray,
    gt: np.ndarray,
    spatial: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spatial coords for pred/gt and a ``(N, K)`` validity mask.

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(N, K, C)``.
    spatial:
        Number of spatial dimensions to compare (``2`` or ``3``). If ``None``
        it is inferred: a trailing visibility channel is assumed when
        ``C in (3, 4)`` *and* both arrays share that ``C`` -- specifically
        ``C == 3`` is treated as 2D+vis, ``C == 4`` as 3D+vis, while ``C == 2``
        is plain 2D. The GT visibility channel (if any) drives the mask.
    """
    c = gt.shape[-1]
    if spatial is None:
        if c == 2:
            spatial, has_vis = 2, False
        elif c == 3:
            spatial, has_vis = 2, True
        elif c == 4:
            spatial, has_vis = 3, True
        else:
            raise ValueError(f"Unsupported coordinate dimension C={c}")
    else:
        has_vis = c > spatial

    pred_xy = pred[..., :spatial]
    gt_xy = gt[..., :spatial]
    if has_vis:
        mask = gt[..., -1] > 0
    else:
        mask = np.ones(gt.shape[:2], dtype=bool)
    return pred_xy, gt_xy, mask


def pck(
    pred: Any,
    gt: Any,
    thr: float = 0.05,
    norm: Any = None,
    *,
    mask: Any = None,
    spatial: Optional[int] = None,
) -> dict[str, Any]:
    """Percentage of Correct Keypoints.

    A keypoint is counted correct when its Euclidean distance to the ground
    truth, divided by a per-sample normalizer, is **strictly less than**
    ``thr``.

    Parameters
    ----------
    pred, gt:
        Predicted / ground-truth keypoints, ``(N, K, C)`` or ``(K, C)``.
    thr:
        Normalized distance threshold (e.g. ``0.05`` for PCK@0.05, ``0.5`` for
        PCKh@0.5 when ``norm`` is the head size).
    norm:
        Per-sample normalizer. Scalar, ``(N,)`` array, or ``None`` (=> ``1.0``,
        i.e. raw pixel/metric threshold). Typically the torso diagonal, head
        size, or bbox size.
    mask:
        Optional ``(N, K)`` boolean validity mask. If ``None`` it is derived
        from the GT visibility channel when present, else all-True.
    spatial:
        Force the number of spatial dimensions (2 or 3). Auto-detected by
        default.

    Returns
    -------
    dict
        ``{"per_joint": (K,) float array, "mean": float, "thr": float}``.
        Joints with no valid samples yield ``nan`` in ``per_joint`` and are
        excluded from ``mean``.
    """
    pred = _as_3d(pred)
    gt = _as_3d(gt)
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"pred/gt sample/keypoint mismatch: {pred.shape} vs {gt.shape}"
        )

    pred_xy, gt_xy, vis_mask = _split_coords_vis(pred, gt, spatial)
    if mask is not None:
        vis_mask = vis_mask & np.asarray(mask, dtype=bool)

    n = pred_xy.shape[0]
    if norm is None:
        norm_arr = np.ones(n, dtype=np.float64)
    else:
        norm_arr = np.asarray(norm, dtype=np.float64)
        if norm_arr.ndim == 0:
            norm_arr = np.full(n, float(norm_arr))
        if norm_arr.shape[0] != n:
            raise ValueError(f"norm length {norm_arr.shape[0]} != N {n}")

    dist = np.linalg.norm(pred_xy - gt_xy, axis=-1)  # (N, K)
    norm_safe = np.where(norm_arr <= 0, _EPS, norm_arr)
    ndist = dist / norm_safe[:, None]

    correct = (ndist < thr) & vis_mask  # (N, K)
    valid_per_joint = vis_mask.sum(axis=0)  # (K,)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_joint = np.where(
            valid_per_joint > 0,
            correct.sum(axis=0) / np.maximum(valid_per_joint, 1),
            np.nan,
        )

    total_valid = int(vis_mask.sum())
    mean = float(correct.sum() / total_valid) if total_valid > 0 else float("nan")
    return {"per_joint": per_joint, "mean": mean, "thr": float(thr)}


def oks(
    pred: Any,
    gt: Any,
    areas: Any,
    sigmas: Any = None,
    *,
    spatial: Optional[int] = None,
) -> np.ndarray:
    """COCO-style Object Keypoint Similarity, per sample.

    ``OKS = mean_i exp(-d_i^2 / (2 * s^2 * k_i^2))`` over visible keypoints,
    where ``d_i`` is the Euclidean distance, ``s^2`` the object area, and
    ``k_i`` the per-keypoint falloff constant.

    Parameters
    ----------
    pred, gt:
        Keypoints ``(N, K, C)`` or ``(K, C)``.
    areas:
        Object area (``s^2``, e.g. bbox area) per sample: scalar or ``(N,)``.
    sigmas:
        Per-keypoint constants ``k_i``, shape ``(K,)``. Defaults to a uniform
        ``0.05`` for every keypoint when ``None``.
    spatial:
        Force spatial dimension count. Auto-detected by default.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` OKS values in ``[0, 1]``. Samples with no visible keypoints
        yield ``0.0``.
    """
    pred = _as_3d(pred)
    gt = _as_3d(gt)
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"pred/gt sample/keypoint mismatch: {pred.shape} vs {gt.shape}"
        )

    pred_xy, gt_xy, vis_mask = _split_coords_vis(pred, gt, spatial)
    n, k = pred_xy.shape[0], pred_xy.shape[1]

    if sigmas is None:
        sig = np.full(k, 0.05, dtype=np.float64)
    else:
        sig = np.asarray(sigmas, dtype=np.float64)
        if sig.shape[0] != k:
            raise ValueError(f"sigmas length {sig.shape[0]} != K {k}")

    area_arr = np.asarray(areas, dtype=np.float64)
    if area_arr.ndim == 0:
        area_arr = np.full(n, float(area_arr))
    if area_arr.shape[0] != n:
        raise ValueError(f"areas length {area_arr.shape[0]} != N {n}")

    d2 = np.sum((pred_xy - gt_xy) ** 2, axis=-1)  # (N, K)
    denom = 2.0 * area_arr[:, None] * (sig[None, :] ** 2) + _EPS
    e = d2 / denom
    similarity = np.exp(-e)  # (N, K)

    similarity = np.where(vis_mask, similarity, 0.0)
    n_vis = vis_mask.sum(axis=1)  # (N,)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n_vis > 0, similarity.sum(axis=1) / np.maximum(n_vis, 1), 0.0)
    return out.astype(np.float64)


def mean_per_joint_error(
    pred: Any,
    gt: Any,
    *,
    mask: Any = None,
    spatial: Optional[int] = None,
) -> dict[str, Any]:
    """Mean Per-Joint Position Error (MPJPE) / mean Euclidean error.

    Parameters
    ----------
    pred, gt:
        Keypoints ``(N, K, C)`` or ``(K, C)``.
    mask:
        Optional ``(N, K)`` validity mask (combined with GT visibility).
    spatial:
        Force spatial dimension count. Auto-detected by default.

    Returns
    -------
    dict
        ``{"per_joint": (K,) array, "mean": float}`` in input coordinate units.
        Joints / overall means over zero valid samples yield ``nan``.
    """
    pred = _as_3d(pred)
    gt = _as_3d(gt)
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"pred/gt sample/keypoint mismatch: {pred.shape} vs {gt.shape}"
        )

    pred_xy, gt_xy, vis_mask = _split_coords_vis(pred, gt, spatial)
    if mask is not None:
        vis_mask = vis_mask & np.asarray(mask, dtype=bool)

    dist = np.linalg.norm(pred_xy - gt_xy, axis=-1)  # (N, K)
    valid_per_joint = vis_mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_joint = np.where(
            valid_per_joint > 0,
            np.where(vis_mask, dist, 0.0).sum(axis=0) / np.maximum(valid_per_joint, 1),
            np.nan,
        )
    total_valid = int(vis_mask.sum())
    mean = (
        float(np.where(vis_mask, dist, 0.0).sum() / total_valid)
        if total_valid > 0
        else float("nan")
    )
    return {"per_joint": per_joint, "mean": mean}
