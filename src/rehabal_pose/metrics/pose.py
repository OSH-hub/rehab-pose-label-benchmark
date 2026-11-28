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

    dist = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    norm_safe = np.where(norm_arr <= 0, _EPS, norm_arr)
    ndist = dist / norm_safe[:, None]

    correct = (ndist < thr) & vis_mask
    valid_per_joint = vis_mask.sum(axis=0)
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

    d2 = np.sum((pred_xy - gt_xy) ** 2, axis=-1)
    denom = 2.0 * area_arr[:, None] * (sig[None, :] ** 2) + _EPS
    e = d2 / denom
    similarity = np.exp(-e)

    similarity = np.where(vis_mask, similarity, 0.0)
    n_vis = vis_mask.sum(axis=1)
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
    pred = _as_3d(pred)
    gt = _as_3d(gt)
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"pred/gt sample/keypoint mismatch: {pred.shape} vs {gt.shape}"
        )

    pred_xy, gt_xy, vis_mask = _split_coords_vis(pred, gt, spatial)
    if mask is not None:
        vis_mask = vis_mask & np.asarray(mask, dtype=bool)

    dist = np.linalg.norm(pred_xy - gt_xy, axis=-1)
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
