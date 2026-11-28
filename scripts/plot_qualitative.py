"""Usage: KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 python scripts/plot_qualitative.py"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

import sys as _sys
_HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.data.skeletons import get_skeleton  # noqa: E402
from rehabal_pose.utils.logging import get_logger  # noqa: E402

from _figstyle import COND_STYLE, save_fig  # noqa: E402

_LOG = get_logger("plot_qualitative")

_MODELS = {
    "A": "outputs/experiments/clinician_i4_uniform30/A_s0/round_3/run/train/weights/best.pt",
    "B": "outputs/experiments/clinician_i4_uniform30/B_s0_sig0/round_3/run/train/weights/best.pt",
    "ref": "outputs/experiments/clinician_i4_uniform30/ref_s0/round_3/run/train/weights/best.pt",
}
_COND_TITLE = {
    "A": "A (pseudo-label)",
    "B": "B (simulated correction)",
    "ref": "ref (full GT)",
}


def _resolve_weights(path: str) -> Optional[str]:
    p = Path(path)
    if p.is_file():
        return str(p)
    if "round_" in str(p):
        for r in range(3, -1, -1):
            cand = Path(str(p).replace("round_3", f"round_{r}"))
            if cand.is_file():
                _LOG.warning("Weights %s missing; falling back to %s", path, cand)
                return str(cand)
    return None


def _load_gt(labels_dir: Path, stem: str, width: float, height: float) -> Optional[np.ndarray]:
    txt = labels_dir / f"{stem}.txt"
    if not txt.is_file():
        return None
    try:
        line = txt.read_text(encoding="utf-8").strip().split("\n")[0]
        vals = np.fromstring(line, sep=" ", dtype=np.float64)
        kp = vals[5:]
        K = len(kp) // 3
        kp = kp.reshape(K, 3)
        kp[:, 0] *= float(width)
        kp[:, 1] *= float(height)
        return kp
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Could not parse GT %s: %s", txt, exc)
        return None


def _joint_error_px(pred: np.ndarray, gt: np.ndarray) -> float:
    vis = gt[:, 2] > 0
    if not np.any(vis):
        return float("nan")
    d = np.linalg.norm(pred[vis, :2] - gt[vis, :2], axis=1)
    return float(np.mean(d))


def _infer_all(weights: str, imgs: list[Path]) -> dict[str, np.ndarray]:
    from rehabal_pose.models.yolo_pose import YoloPoseModel  # noqa: PLC0415

    model = YoloPoseModel({"workers": 0})
    model.load(weights)

    out: dict[str, np.ndarray] = {}
    CHUNK = 256
    for start in range(0, len(imgs), CHUNK):
        chunk = imgs[start:start + CHUNK]
        for p in model.predict([str(c) for c in chunk]):
            kp = np.asarray(p.keypoints, dtype=np.float64)
            if kp.shape == (26, 3):
                out[p.sample_id] = kp

    del model
    try:
        import gc; gc.collect()  # noqa: E702
        import torch; torch.cuda.empty_cache()  # type: ignore  # noqa: E702
    except Exception:  # noqa: BLE001
        pass
    return out


def _select_frames(
    stems: list[str],
    errs: dict[str, dict[str, float]],
    n: int,
) -> list[str]:
    scored: list[tuple[float, str, str]] = []
    for s in stems:
        e = errs.get(s)
        if not e or any(np.isnan(v) for v in e.values()):
            continue
        if e["B"] > 60.0 or e["ref"] > 60.0:
            continue
        score = e["A"] - 0.5 * (e["B"] + e["ref"])
        clip = s.rsplit("_f", 1)[0]
        scored.append((score, clip, s))
    scored.sort(key=lambda t: t[0], reverse=True)

    picked: list[str] = []
    used_clips: set[str] = set()
    for _, clip, s in scored:
        if clip in used_clips:
            continue
        picked.append(s)
        used_clips.add(clip)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for _, _clip, s in scored:
            if s not in picked:
                picked.append(s)
            if len(picked) >= n:
                break
    return picked


def _draw_skeleton(ax, kp: np.ndarray, edges, color: str, *, vis_mask=None,
                   lw: float = 2.0, ms: float = 26.0, alpha: float = 1.0,
                   zorder: int = 3) -> None:
    if vis_mask is None:
        vis_mask = np.ones(kp.shape[0], dtype=bool)
    for i, j in edges:
        if i < kp.shape[0] and j < kp.shape[0] and vis_mask[i] and vis_mask[j]:
            ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]],
                    color=color, lw=lw, alpha=alpha, zorder=zorder,
                    solid_capstyle="round")
    m = vis_mask
    ax.scatter(kp[m, 0], kp[m, 1], s=ms, c=color, alpha=alpha,
               zorder=zorder + 1, edgecolors="none")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-dir", default="outputs/rehab24_6_yolo/images/val")
    ap.add_argument("--labels-dir", default="outputs/rehab24_6_yolo/labels/val")
    ap.add_argument("--out", default="outputs/experiments/paper_figures/fig_qualitative.png")
    ap.add_argument("--n-frames", type=int, default=4)
    args = ap.parse_args(argv)

    val_dir = Path(args.val_dir)
    labels_dir = Path(args.labels_dir)
    imgs = sorted(val_dir.glob("*.jpg"))
    if not imgs:
        _LOG.error("No val images under %s", val_dir)
        return 1
    _LOG.info("Found %d val images", len(imgs))

    from PIL import Image  # noqa: PLC0415
    size_by_stem: dict[str, tuple[float, float]] = {}
    for ip in imgs:
        with Image.open(ip) as im:
            size_by_stem[ip.stem] = (float(im.width), float(im.height))

    weights: dict[str, str] = {}
    for cond, path in _MODELS.items():
        w = _resolve_weights(path)
        if w is None:
            _LOG.error("No weights found for condition %s (%s)", cond, path)
            return 1
        weights[cond] = w
        _LOG.info("Condition %s -> %s", cond, w)

    preds: dict[str, dict[str, np.ndarray]] = {}
    for cond, w in weights.items():
        _LOG.info("Inferring val with condition %s ...", cond)
        preds[cond] = _infer_all(w, imgs)
        _LOG.info("  condition %s: %d/%d frames predicted", cond, len(preds[cond]), len(imgs))

    gt_by_stem: dict[str, np.ndarray] = {}
    for ip in imgs:
        w, h = size_by_stem[ip.stem]
        g = _load_gt(labels_dir, ip.stem, w, h)
        if g is not None:
            gt_by_stem[ip.stem] = g

    errs: dict[str, dict[str, float]] = {}
    for ip in imgs:
        stem = ip.stem
        gt = gt_by_stem.get(stem)
        if gt is None:
            continue
        row: dict[str, float] = {}
        ok = True
        for cond in _MODELS:
            pr = preds[cond].get(stem)
            if pr is None:
                ok = False
                break
            row[cond] = _joint_error_px(pr, gt)
        if ok:
            errs[stem] = row

    _LOG.info("Frames with GT + all-3 predictions: %d", len(errs))
    frames = _select_frames([ip.stem for ip in imgs], errs, args.n_frames)
    if not frames:
        _LOG.error("No frames selected (no frame had GT + all 3 predictions).")
        return 1
    for s in frames:
        e = errs[s]
        _LOG.info("Selected %s  A=%.1fpx B=%.1fpx ref=%.1fpx", s, e["A"], e["B"], e["ref"])

    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    edges = get_skeleton("rehab26").edges
    gt_gray = "#9a9a9a"

    nrows = len(frames)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.05))
    axes = np.atleast_2d(axes)

    col_titles = ["GT skeleton", _COND_TITLE["A"], _COND_TITLE["B"], _COND_TITLE["ref"]]

    for r, stem in enumerate(frames):
        gt = gt_by_stem[stem]
        gt_vis = gt[:, 2] > 0
        frame_w, frame_h = size_by_stem[stem]

        for c in range(ncols):
            ax = axes[r, c]
            ax.set_facecolor("white")
            ax.set_xlim(0, frame_w)
            ax.set_ylim(frame_h, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

            _draw_skeleton(ax, gt, edges, gt_gray, vis_mask=gt_vis,
                           lw=1.6, ms=14, alpha=0.55, zorder=2)

            if c == 0:
                if r == 0:
                    ax.set_title(col_titles[c], fontsize=11)
                ax.set_ylabel(f"Pose {r + 1}", fontsize=9)
                continue

            cond = ["A", "B", "ref"][c - 1]
            pr = preds[cond][stem]
            col = COND_STYLE[cond]["color"]
            _draw_skeleton(ax, pr, edges, col, lw=2.0, ms=22, alpha=0.95, zorder=4)

            e = errs[stem][cond]
            title = f"{col_titles[c]}\nerr {e:.0f}px" if r == 0 else f"err {e:.0f}px"
            ax.set_title(title, fontsize=10, color=col)

    handles = [
        Line2D([0], [0], color=gt_gray, lw=2, marker="o", label="Ground truth"),
        Line2D([0], [0], color=COND_STYLE["A"]["color"], lw=2, marker="o", label="A prediction"),
        Line2D([0], [0], color=COND_STYLE["B"]["color"], lw=2, marker="o", label="B prediction"),
        Line2D([0], [0], color=COND_STYLE["ref"]["color"], lw=2, marker="o", label="ref prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = save_fig(plt, args.out)
    plt.close(fig)
    _LOG.info("Wrote %s (+ .pdf)", out)
    print(f"OK wrote {out} and {Path(out).with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
