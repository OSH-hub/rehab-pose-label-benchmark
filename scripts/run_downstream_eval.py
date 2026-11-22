"""Per-budget downstream rehabilitation-assessment evaluation.

Given the per-budget ``best.pt`` paths from a completed AL experiment session
(condition A / B / ref), this script:

1. Runs YOLO-Pose inference on the fixed val set using each budget's weights to
   obtain predicted keypoint sequences (one per image).
2. Loads the corresponding GT keypoint sequences from the YOLO-pose label files.
3. Groups sequences by recording and orders them by numeric source-frame index
   to produce ``(T, K, 3)`` arrays.
4. Calls :func:`rehabal_pose.pipelines.evaluate_rehab.evaluate_rehab` per clip
   and aggregates joint-angle MAE / ROM error / peak-angle error / smoothness
   error into a per-budget metrics JSON.

Output: ``outputs/experiments/downstream/<session_tag>_downstream.json``

Import-safety requirements:
  - No torch / ultralytics / cv2 at module load (all guarded inside functions).
  - Mock-safe: importing this module must not trigger GPU work.

The manuscript's downstream numbers come from the uniform-30-epoch sessions
(``outputs/experiments/downstream_*_u30``); ``scripts/run_downstream_u30.sh``
drives the three backbone tracks. Passing ``assessment_labels_path`` from the
REHAB24-6 Segmentation.csv additionally enables correctness-F1, which the
manuscript does not report.

Usage (GPU required)::

    python scripts/run_downstream_eval.py \\
        --weights-glob "outputs/experiments/clinician/A_s0/round_*/train/weights/best.pt" \\
        --val-dir outputs/rehab24_6_yolo/images/val \\
        --labels-dir outputs/rehab24_6_yolo/labels/val \\
        --out outputs/experiments/downstream/A_s0_downstream.json

    # Or specify weights explicitly per budget:
    python scripts/run_downstream_eval.py \\
        --budgets 0.05 0.10 0.20 0.40 \\
        --weights-list path_b005/best.pt path_b010/best.pt path_b020/best.pt path_b040/best.pt \\
        --val-dir outputs/rehab24_6_yolo/images/val \\
        --labels-dir outputs/rehab24_6_yolo/labels/val \\
        --out outputs/experiments/downstream/A_s0_downstream.json

NOTE (budget labelling): ``--weights-glob`` auto-infers each budget from a
  ``b<frac>`` token in the weights' grandparent directory name. The clinician
  sessions use ``round_<r>`` directories (no ``b<frac>`` token), so the glob
  form would mislabel budgets as 0, 1, 2, 3. For those sessions use
  ``--weights-list`` with an explicit ``--budgets 0.05 0.1 0.2 0.4``.

NOTE (coordinate frame): YOLO-pose predictions come back in absolute pixel
  coordinates while the GT YOLO-pose label files are normalized to [0, 1].
  Both are mapped to an isotropically scaled image frame by dividing pixel
  coordinates by the image diagonal. This preserves angles on non-square
  images and gives smoothness a resolution-independent spatial scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import sys as _sys
_HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.pipelines.evaluate_rehab import evaluate_rehab, REHAB26_JOINT_TRIPLETS  # noqa: E402
from rehabal_pose.utils.io import ensure_dir, write_json  # noqa: E402
from rehabal_pose.utils.logging import get_logger  # noqa: E402

_LOG = get_logger("downstream_eval")


# ---------------------------------------------------------------------------
# Helpers (all GPU-free)
# ---------------------------------------------------------------------------


def _load_gt_keypoints(labels_dir: Path, stems: list[str]) -> dict[str, np.ndarray]:
    """Load GT keypoints from YOLO-pose .txt files for the given stems.

    YOLO-pose label format (normalized): one row per instance:
      ``<class_id> <cx> <cy> <w> <h> <x1> <y1> <v1> ... <xK> <yK> <vK>``
    Returns dict stem -> ``(K, 3)`` float64 array ``[x, y, v]`` in YOLO's
    width/height-normalized frame. The caller converts it to the shared
    isotropically scaled image frame before computing geometry.
    Returns an empty dict entry if the label file is missing / unparseable.
    """
    out: dict[str, np.ndarray] = {}
    for stem in stems:
        txt = labels_dir / f"{stem}.txt"
        if not txt.is_file():
            _LOG.debug("GT label missing: %s", txt)
            out[stem] = np.zeros((26, 3), dtype=np.float64)
            continue
        try:
            vals = np.fromstring(txt.read_text(encoding="utf-8").strip().split("\n")[0],
                                 sep=" ", dtype=np.float64)
            # cols: class cx cy w h kp1x kp1y kp1v ... kpKx kpKy kpKv
            kp_flat = vals[5:]
            K = len(kp_flat) // 3
            kps = kp_flat.reshape(K, 3)[:, :3]
            out[stem] = kps.astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Could not parse GT label %s: %s", txt, exc)
            out[stem] = np.zeros((26, 3), dtype=np.float64)
    return out


def _pixel_keypoints_to_isotropic(
    kp: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Map pixel keypoints to image-diagonal units without changing geometry."""
    out = np.asarray(kp, dtype=np.float64).copy()
    if not (
        width
        and height
        and np.isfinite(width)
        and np.isfinite(height)
        and width > 0
        and height > 0
    ):
        raise ValueError(f"invalid image size: width={width!r}, height={height!r}")
    scale = float(np.hypot(width, height))
    out[:, 0] = out[:, 0] / scale
    out[:, 1] = out[:, 1] / scale
    return out


def _yolo_keypoints_to_isotropic(
    kp: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Map YOLO-normalized keypoints to image-diagonal units."""
    out = np.asarray(kp, dtype=np.float64).copy()
    if not (
        width
        and height
        and np.isfinite(width)
        and np.isfinite(height)
        and width > 0
        and height > 0
    ):
        raise ValueError(f"invalid image size: width={width!r}, height={height!r}")
    scale = float(np.hypot(width, height))
    out[:, 0] = out[:, 0] * float(width) / scale
    out[:, 1] = out[:, 1] * float(height) / scale
    return out


def _image_sizes(imgs: list[Path]) -> dict[str, tuple[float, float]]:
    """Map image stem -> (width, height) in pixels, reading only headers.

    Uses Pillow (a transitive ultralytics dependency); imported lazily so the
    module stays free of heavy/GPU imports at load time.
    """
    sizes: dict[str, tuple[float, float]] = {}
    try:
        from PIL import Image  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Pillow unavailable (%s); predictions left in pixel frame.", exc)
        return sizes
    for ip in imgs:
        try:
            with Image.open(ip) as im:
                sizes[ip.stem] = (float(im.width), float(im.height))
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Could not read image size for %s: %s", ip, exc)
    return sizes


def _infer_val_keypoints(weights: str, val_img_dir: Path,
                          batch: int = 16) -> dict[str, np.ndarray]:
    """Run YOLO-Pose inference on val images, return stem -> (K,3) predictions.

    Predictions remain in absolute pixel coordinates. The evaluation caller
    maps them and the normalized GT labels to a shared isotropic frame.

    Heavy imports are inside this function (GPU path). Do NOT call at module load.
    """
    # Keep heavyweight dependencies out of the module-import path.
    from rehabal_pose.models.yolo_pose import YoloPoseModel  # noqa: PLC0415

    imgs = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    if not imgs:
        _LOG.warning("No images found under %s", val_img_dir)
        return {}

    _LOG.info("Inferring %d val images with weights %s", len(imgs), weights)
    model = YoloPoseModel({"batch": batch, "workers": 0})
    model.load(weights)

    out: dict[str, np.ndarray] = {}
    CHUNK = 256
    for start in range(0, len(imgs), CHUNK):
        chunk = imgs[start:start + CHUNK]
        preds = model.predict([str(p) for p in chunk])
        for p in preds:
            if p.keypoints is not None:
                kp = np.asarray(p.keypoints, dtype=np.float64)
                if kp.shape == (26, 3):
                    out[p.sample_id] = kp

    # Ensure VRAM released after each budget's inference pass.
    del model
    try:
        import gc; gc.collect()  # noqa: E702
        import torch; torch.cuda.empty_cache()  # type: ignore  # noqa: E702
    except Exception:  # noqa: BLE001
        pass

    _LOG.info("Got predictions for %d / %d val images", len(out), len(imgs))
    return out


def _stems_from_dir(img_dir: Path) -> list[str]:
    """Return sorted image stems from a directory."""
    return sorted(p.stem for p in img_dir.glob("*.jpg")) + \
           sorted(p.stem for p in img_dir.glob("*.png"))


def _group_by_clip(stems: list[str]) -> dict[str, list[str]]:
    """Group image stems by recording/clip prefix (everything before ``_rep``).

    Example: ``PM_045_rep00_f100`` -> clip ``PM_045_rep00``.
    Falls back to using the full stem as its own clip if the pattern is absent.
    """
    import re
    pat = re.compile(r"^(?P<clip>.+?)_f\d+$")
    clips: dict[str, list[str]] = {}
    for s in stems:
        m = pat.match(s)
        key = m.group("clip") if m else s
        clips.setdefault(key, []).append(s)
    def _frame_number(stem: str) -> int:
        match = pat.match(stem)
        return int(stem.rsplit("_f", 1)[1]) if match else 0

    return {k: sorted(v, key=_frame_number) for k, v in clips.items()}


def _eval_budget(
    weights: str,
    val_img_dir: Path,
    gt_labels_dir: Path,
    *,
    joint_triplets: Optional[dict] = None,
    frame_interval_sec: float = 20.0 / 30.0,
) -> dict[str, Any]:
    """Evaluate downstream rehab metrics for one budget's best.pt.

    Returns a metrics dict with per-clip and aggregate results.
    GPU work is done inside _infer_val_keypoints (guarded import).
    """
    stems = _stems_from_dir(val_img_dir)
    if not stems:
        return {"error": f"no images in {val_img_dir}", "n_clips": 0}

    pred_pixels = _infer_val_keypoints(weights, val_img_dir)
    gt_yolo = _load_gt_keypoints(gt_labels_dir, stems)
    image_paths = sorted(val_img_dir.glob("*.jpg")) + sorted(val_img_dir.glob("*.png"))
    size_by_stem = _image_sizes(image_paths)
    missing_sizes = sorted(set(stems) - set(size_by_stem))
    if missing_sizes:
        preview = ", ".join(missing_sizes[:3])
        raise RuntimeError(
            f"image dimensions unavailable for {len(missing_sizes)} frame(s): {preview}"
        )

    pred_by_stem: dict[str, np.ndarray] = {}
    gt_by_stem: dict[str, np.ndarray] = {}
    for stem in stems:
        width, height = size_by_stem[stem]
        if stem in pred_pixels:
            pred_by_stem[stem] = _pixel_keypoints_to_isotropic(
                pred_pixels[stem], width, height
            )
        if stem in gt_yolo:
            gt_by_stem[stem] = _yolo_keypoints_to_isotropic(
                gt_yolo[stem], width, height
            )
    clips = _group_by_clip(stems)
    triplets = joint_triplets or REHAB26_JOINT_TRIPLETS

    clip_results: list[dict] = []
    for clip_id, clip_stems in clips.items():
        # Build (T, K, 3) sequences — pad missing predictions with zeros
        pred_seq = np.stack(
            [pred_by_stem.get(s, np.zeros((26, 3), dtype=np.float64))
             for s in clip_stems], axis=0
        )
        gt_seq = np.stack(
            [gt_by_stem.get(s, np.zeros((26, 3), dtype=np.float64))
             for s in clip_stems], axis=0
        )
        if pred_seq.shape[0] < 1:
            continue
        # Use only the spatial (x, y) columns. ``joint_angle`` treats every
        # supplied column as a coordinate axis, so leaving the visibility column
        # in (pred = YOLO confidence ~[0,1] vs GT = label flag {0,1,2}) would
        # inject a phantom third axis and corrupt the joint angles. Both pred and
        # GT x/y are in the same image-diagonal-normalized frame at this point.
        pred_xy = pred_seq[..., :2]
        gt_xy = gt_seq[..., :2]
        try:
            m = evaluate_rehab(
                pred_xy,
                gt_xy,
                joint_triplets=triplets,
                dt=frame_interval_sec,
            )
            m["clip_id"] = clip_id
            m["n_frames"] = int(pred_seq.shape[0])
            clip_results.append(m)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("evaluate_rehab failed for clip %s: %s", clip_id, exc)

    if not clip_results:
        return {"error": "no clips evaluated", "n_clips": 0}

    # Aggregate per-joint angle MAE (mean over clips)
    angle_mae_keys = list(triplets.keys()) + ["mean"]
    agg_mae: dict[str, list[float]] = {k: [] for k in angle_mae_keys}
    agg_rom: dict[str, list[float]] = {k: [] for k in angle_mae_keys}
    smoothness_vals: list[float] = []

    for cr in clip_results:
        for k in angle_mae_keys:
            v = cr.get("joint_angle_mae", {}).get(k, float("nan"))
            if not np.isnan(v):
                agg_mae[k].append(v)
            vr = cr.get("rom_error", {}).get(k, float("nan"))
            if not np.isnan(vr):
                agg_rom[k].append(vr)
        s = cr.get("smoothness_error", float("nan"))
        if not np.isnan(s):
            smoothness_vals.append(s)

    def _nanmean(lst: list[float]) -> float:
        return float(np.nanmean(lst)) if lst else float("nan")

    aggregate = {
        "n_clips": len(clip_results),
        "n_frames_total": sum(c["n_frames"] for c in clip_results),
        "joint_angle_mae_mean": {k: _nanmean(agg_mae[k]) for k in angle_mae_keys},
        "rom_error_mean": {k: _nanmean(agg_rom[k]) for k in angle_mae_keys},
        "smoothness_error_mean": _nanmean(smoothness_vals),
        "n_pred_found": len(pred_by_stem),
        "n_stems": len(stems),
        "coordinate_frame": "image-diagonal-normalized",
        "frame_interval_sec": float(frame_interval_sec),
    }
    return {
        "aggregate": aggregate,
        "per_clip": clip_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: per-budget downstream rehab eval (needs a GPU)."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    weights_group = ap.add_mutually_exclusive_group(required=True)
    weights_group.add_argument(
        "--weights-glob",
        metavar="GLOB",
        help="Glob pattern for best.pt files, sorted to infer budget order. "
             "Example: 'outputs/experiments/clinician/A_s0/round_*/run/train/weights/best.pt'",
    )
    weights_group.add_argument(
        "--weights-list",
        nargs="+",
        metavar="PT",
        help="Explicit list of best.pt paths, one per budget (order matches --budgets).",
    )
    ap.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=None,
        help="Budget fraction for each weight file (e.g. 0.05 0.10 0.20 0.40). "
             "Required with --weights-list; ignored with --weights-glob (auto-inferred).",
    )
    ap.add_argument(
        "--val-dir",
        required=True,
        help="Directory of val images (YOLO-pose dataset val split).",
    )
    ap.add_argument(
        "--labels-dir",
        required=True,
        help="Directory of val GT label .txt files (YOLO-pose format).",
    )
    ap.add_argument(
        "--out",
        default="outputs/experiments/downstream/downstream.json",
        help="Output JSON path for aggregated per-budget metrics.",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Inference batch size (default 16).",
    )
    ap.add_argument(
        "--frame-interval-sec",
        type=float,
        default=20.0 / 30.0,
        help="Seconds between sampled validation frames (default 20/30 for the "
             "reported REHAB24-6 preprocessing).",
    )
    args = ap.parse_args(argv)

    val_img_dir = Path(args.val_dir)
    gt_labels_dir = Path(args.labels_dir)
    out_path = Path(args.out)

    if not val_img_dir.is_dir():
        _LOG.error("val-dir not found: %s", val_img_dir)
        return 1
    if not gt_labels_dir.is_dir():
        _LOG.error("labels-dir not found: %s", gt_labels_dir)
        return 1

    # Resolve weights list
    if args.weights_list:
        weights_paths = [str(Path(p)) for p in args.weights_list]
        if args.budgets:
            if len(args.budgets) != len(weights_paths):
                _LOG.error("--budgets length (%d) != --weights-list length (%d)",
                           len(args.budgets), len(weights_paths))
                return 1
            budgets = args.budgets
        else:
            budgets = list(range(len(weights_paths)))
    else:
        import glob as _glob
        weights_paths = sorted(_glob.glob(args.weights_glob))
        if not weights_paths:
            _LOG.error("No files matched glob: %s", args.weights_glob)
            return 1
        # Auto-infer budget from directory name (expects round_<r> or b<budget>)
        import re
        budgets = []
        for wp in weights_paths:
            m = re.search(r"b([\d.]+)", str(Path(wp).parent.parent.parent.name))
            if m:
                budgets.append(float(m.group(1)))
            else:
                budgets.append(float(len(budgets)))

    _LOG.info("Running per-budget downstream eval: %d budgets, val=%s",
              len(budgets), val_img_dir)

    per_budget: dict[str, Any] = {}
    had_failure = False
    for budget, wp in zip(budgets, weights_paths):
        bkey = f"budget_{budget:g}"
        _LOG.info("Budget %g: %s", budget, wp)
        if not Path(wp).is_file():
            _LOG.warning("  weights file missing: %s — skipping", wp)
            per_budget[bkey] = {"error": f"weights not found: {wp}"}
            had_failure = True
            continue
        try:
            result = _eval_budget(
                wp,
                val_img_dir,
                gt_labels_dir,
                frame_interval_sec=args.frame_interval_sec,
            )
            per_budget[bkey] = {"budget": budget, "weights": wp, **result}
            agg = result.get("aggregate", {})
            mae_mean = agg.get("joint_angle_mae_mean", {}).get("mean", float("nan"))
            rom_mean = agg.get("rom_error_mean", {}).get("mean", float("nan"))
            _LOG.info("  budget %g: clips=%d angle_MAE_mean=%.3f ROM_err_mean=%.3f",
                      budget, agg.get("n_clips", 0), mae_mean, rom_mean)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("  budget %g FAILED: %s", budget, exc)
            per_budget[bkey] = {"budget": budget, "weights": wp, "error": str(exc)}
            had_failure = True

    output = {
        "budgets": per_budget,
        "n_budgets": len(budgets),
        "coordinate_frame": "image-diagonal-normalized",
        "frame_interval_sec": float(args.frame_interval_sec),
    }
    ensure_dir(out_path.parent)
    write_json(out_path, output)
    _LOG.info("Wrote downstream metrics -> %s", out_path)
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
