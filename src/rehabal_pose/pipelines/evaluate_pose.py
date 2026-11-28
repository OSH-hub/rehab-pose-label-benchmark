from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from ..metrics.pose import mean_per_joint_error, oks, pck
from ..utils.io import write_json
from ..utils.logging import get_logger

__all__ = [
    "evaluate_pose",
    "evaluate_from_manifests",
    "main",
]

_LOG = get_logger(__name__)

_OKS_THRESHOLDS: tuple[float, ...] = tuple(
    round(float(t), 2) for t in np.arange(0.50, 1.00, 0.05)
)


def _as_3d(arr: Any) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(
            f"Expected keypoints of shape (N, K, C) or (K, C), got ndim={a.ndim}"
        )
    return a


def _vis_mask(gt: np.ndarray) -> np.ndarray:
    c = gt.shape[-1]
    if c in (3, 4):
        return gt[..., -1] > 0
    return np.ones(gt.shape[:2], dtype=bool)


def _derive_areas(gt: np.ndarray) -> np.ndarray:
    gt = _as_3d(gt)
    xy = gt[..., :2]
    vis = _vis_mask(gt)
    areas = np.empty(xy.shape[0], dtype=np.float64)
    for i in range(xy.shape[0]):
        m = vis[i]
        pts = xy[i][m] if bool(m.any()) else xy[i]
        if pts.shape[0] == 0:
            areas[i] = 1.0
            continue
        wh = pts.max(axis=0) - pts.min(axis=0)
        a = float(wh[0] * wh[1])
        areas[i] = a if a > 0.0 else 1.0
    return areas


def evaluate_pose(
    pred: Any,
    gt: Any,
    *,
    thresholds: Sequence[float] = (0.05, 0.1, 0.2),
    norm: Any = None,
    sigmas: Any = None,
    areas: Any = None,
) -> dict[str, Any]:
    pred = _as_3d(pred)
    gt = _as_3d(gt)
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(
            f"pred/gt sample/keypoint mismatch: {pred.shape} vs {gt.shape}"
        )
    n, k = int(pred.shape[0]), int(pred.shape[1])

    pck_mean: dict[str, float] = {}
    pck_per_joint: dict[str, list[float]] = {}
    for thr in thresholds:
        res = pck(pred, gt, float(thr), norm)
        key = _fmt_thr(thr)
        pck_mean[key] = res["mean"]
        pck_per_joint[key] = [float(v) for v in np.asarray(res["per_joint"])]

    area_arr = _derive_areas(gt) if areas is None else areas
    oks_vals = oks(pred, gt, area_arr, sigmas)
    if oks_vals.size > 0:
        oks_mean = float(np.mean(oks_vals))
        ap_per_thr = {
            f"{t:.2f}": float(np.mean(oks_vals >= t)) for t in _OKS_THRESHOLDS
        }
        oks_map = float(np.mean(list(ap_per_thr.values())))
    else:
        oks_mean = float("nan")
        ap_per_thr = {f"{t:.2f}": float("nan") for t in _OKS_THRESHOLDS}
        oks_map = float("nan")

    mpjpe = mean_per_joint_error(pred, gt)

    return {
        "n_samples": n,
        "num_keypoints": k,
        "pck": pck_mean,
        "pck_per_joint": pck_per_joint,
        "oks": {
            "mean": oks_mean,
            "mAP": oks_map,
            "AP": ap_per_thr,
            "derived_areas": areas is None,
        },
        "mpjpe": {
            "mean": mpjpe["mean"],
            "per_joint": [float(v) for v in np.asarray(mpjpe["per_joint"])],
        },
    }


def _fmt_thr(thr: float) -> str:
    s = f"{float(thr):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _manifest_records(manifest: Any) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    to_dict = getattr(manifest, "to_dict", None)
    if to_dict is not None and hasattr(manifest, "columns"):
        return list(manifest.to_dict("records"))
    if isinstance(manifest, dict):
        recs: list[dict[str, Any]] = []
        for sid, val in manifest.items():
            if isinstance(val, dict):
                recs.append({"sample_id": sid, **val})
            else:
                recs.append({"sample_id": sid, "exercise": val})
        return recs
    return [dict(r) for r in manifest]


def _build_arrays(
    sample_ids: Sequence[str],
    kp_loader: Callable[[str], np.ndarray],
) -> np.ndarray:
    rows = [np.asarray(kp_loader(sid), dtype=np.float64) for sid in sample_ids]
    return np.stack(rows, axis=0)


def evaluate_from_manifests(
    pred_manifest: Any,
    gt_manifest: Any,
    kp_loader_pred: Callable[[str], np.ndarray],
    kp_loader_gt: Callable[[str], np.ndarray],
    *,
    thresholds: Sequence[float] = (0.05, 0.1, 0.2),
    norm: Any = None,
    sigmas: Any = None,
    areas: Any = None,
    exercise_key: str = "exercise",
) -> dict[str, Any]:
    pred_recs = _manifest_records(pred_manifest)
    gt_recs = _manifest_records(gt_manifest)

    pred_ids = {str(r["sample_id"]) for r in pred_recs if r.get("sample_id") is not None}
    gt_by_id = {
        str(r["sample_id"]): r for r in gt_recs if r.get("sample_id") is not None
    }

    aligned: list[str] = [sid for sid in gt_by_id if sid in pred_ids]
    if not aligned:
        _LOG.warning("evaluate_from_manifests: no overlapping sample_ids")
        return {
            "n_aligned": 0,
            "sample_ids": [],
            "overall": None,
            "per_exercise": {},
        }

    pred_arr = _build_arrays(aligned, kp_loader_pred)
    gt_arr = _build_arrays(aligned, kp_loader_gt)

    overall = evaluate_pose(
        pred_arr,
        gt_arr,
        thresholds=thresholds,
        norm=_subset_norm(norm, range(len(aligned))),
        sigmas=sigmas,
        areas=_subset_areas(areas, range(len(aligned))),
    )

    groups: dict[str, list[int]] = {}
    for idx, sid in enumerate(aligned):
        ex = gt_by_id[sid].get(exercise_key)
        ex = "unknown" if ex is None else str(ex)
        groups.setdefault(ex, []).append(idx)

    per_exercise: dict[str, Any] = {}
    for ex, idxs in groups.items():
        sel = np.asarray(idxs, dtype=int)
        per_exercise[ex] = evaluate_pose(
            pred_arr[sel],
            gt_arr[sel],
            thresholds=thresholds,
            norm=_subset_norm(norm, idxs),
            sigmas=sigmas,
            areas=_subset_areas(areas, idxs),
        )

    return {
        "n_aligned": len(aligned),
        "sample_ids": aligned,
        "overall": overall,
        "per_exercise": per_exercise,
    }


def _subset_norm(norm: Any, idxs: Iterable[int]) -> Any:
    if norm is None:
        return None
    arr = np.asarray(norm, dtype=np.float64)
    if arr.ndim == 0:
        return norm
    return arr[np.asarray(list(idxs), dtype=int)]


def _subset_areas(areas: Any, idxs: Iterable[int]) -> Any:
    if areas is None:
        return None
    arr = np.asarray(areas, dtype=np.float64)
    if arr.ndim == 0:
        return areas
    return arr[np.asarray(list(idxs), dtype=int)]


def _load_keypoints(path: str | Path) -> np.ndarray:
    p = Path(path)
    obj = np.load(p, allow_pickle=False)
    if hasattr(obj, "files"):
        if not obj.files:
            raise ValueError(f"No arrays found in {p}")
        return np.asarray(obj[obj.files[0]], dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def evaluate_pose_files(
    pred_path: str | Path,
    gt_path: str | Path,
    out_path: str | Path | None = None,
    *,
    thresholds: Sequence[float] = (0.05, 0.1, 0.2),
) -> dict[str, Any]:
    pred = _load_keypoints(pred_path)
    gt = _load_keypoints(gt_path)
    metrics = evaluate_pose(pred, gt, thresholds=thresholds)
    if out_path is not None:
        write_json(out_path, metrics)
        _LOG.info("Wrote pose metrics -> %s", out_path)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate 2D pose metrics (PCK / OKS-mAP / MPJPE)."
    )
    parser.add_argument(
        "--pred", required=True, help="predicted keypoints .npy (N,K,C)"
    )
    parser.add_argument(
        "--gt", required=True, help="ground-truth keypoints .npy (N,K,C)"
    )
    parser.add_argument("--out", help="output metrics .json")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.1,0.2",
        help="comma-separated PCK thresholds (default 0.05,0.1,0.2)",
    )
    args = parser.parse_args(argv)

    thresholds = tuple(float(t) for t in str(args.thresholds).split(",") if t.strip())
    metrics = evaluate_pose_files(
        args.pred, args.gt, args.out, thresholds=thresholds
    )
    if args.out is None:
        _LOG.info("PCK=%s OKS-mAP=%s", metrics["pck"], metrics["oks"]["mAP"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
