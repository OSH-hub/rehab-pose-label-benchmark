"""Image-level label-budget active-learning curve for REHAB24-6.

Reuses the PRE-EXTRACTED YOLO-pose dataset (``outputs/rehab24_6_yolo``) so no
video frames are re-extracted per round. For each acquisition strategy and each
label budget we:

  1. score the unlabeled pool with a base model (trained once on a seed subset),
  2. select ``budget * N`` images via the strategy (``rehabal_pose.al``),
  3. train a FRESH YOLO-pose model on the selected images,
  4. evaluate on the FIXED subject-independent val split -> pose mAP.

Outputs ``curve.csv`` + ``summary.json`` (+ a PNG if matplotlib is present) and
per-strategy AUC under ``outputs/experiments/al_curve/``.

Single-step AL (score-once, select-to-budget) — a faithful first-pass comparison;
iterative multi-round acquisition is a later refinement.

Usage (first curve)::

    python scripts/al_budget_curve.py
    python scripts/al_budget_curve.py --smoke         # 2 strategies, 2 budgets, 1 epoch
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rehabal_pose.al import get_strategy
from rehabal_pose.al.base import Pool
from rehabal_pose.data import ManifestBuilder
from rehabal_pose.data.yolo_dataset import rehab26_flip_idx
from rehabal_pose.metrics.learning_curve import area_under_learning_curve
from rehabal_pose.models.yolo_pose import YoloPoseModel
from rehabal_pose.utils.io import ensure_dir, write_json
from rehabal_pose.utils.logging import get_logger
from rehabal_pose.utils.seed import set_seed

_LOG = get_logger("al_curve")

DATASET = Path("outputs/rehab24_6_yolo")
MANIFEST = Path("outputs/rehab24_6_real/manifest.parquet")
WORK = Path("outputs/experiments/al_curve")

_STEM_RE = re.compile(r"^(?P<rec>.+?)_rep\d+_f(?P<idx>\d+)$")


def _recording_and_frame(stem: str) -> tuple[str, int]:
    m = _STEM_RE.match(stem)
    if m:
        return m.group("rec"), int(m.group("idx"))
    return stem, 0


def _recording_meta(manifest_df: pd.DataFrame) -> dict[str, dict]:
    """video_id -> {subject_id, exercise, disease, camera_id}."""
    meta: dict[str, dict] = {}
    for r in manifest_df.drop_duplicates("video_id").itertuples():
        meta[str(r.video_id)] = {
            "subject_id": getattr(r, "subject_id", None),
            "exercise": getattr(r, "exercise", None),
            "disease": getattr(r, "disease", None),
            "camera_id": getattr(r, "camera_id", None),
        }
    return meta


def _write_round_yaml(out_dir: Path, train_list: Path) -> Path:
    """data.yaml whose train is an image-list txt; val is the fixed val dir."""
    flip = rehab26_flip_idx()
    yaml = (
        f"path: {DATASET.resolve().as_posix()}\n"
        f"train: {train_list.resolve().as_posix()}\n"
        f"val: images/val\n\n"
        f"kpt_shape: [26, 3]\n"
        f"flip_idx: {flip}\n\n"
        f"names:\n  0: person\n"
    )
    p = out_dir / "data.yaml"
    p.write_text(yaml, encoding="utf-8")
    return p


def _build_pool(
    pool_imgs: list[str],
    base_model: YoloPoseModel,
    rec_meta: dict[str, dict],
) -> Pool:
    """Score the pool with the base model -> aligned predictions + embeddings."""
    _LOG.info("Scoring pool (%d images) with the base model...", len(pool_imgs))
    # Batch to avoid "too many open files" on large pools.
    CHUNK = 256
    by_stem = {}
    for s in range(0, len(pool_imgs), CHUNK):
        for p in base_model.predict(pool_imgs[s:s + CHUNK]):
            by_stem[p.sample_id] = p

    N = len(pool_imgs)
    K = 26
    pred_arr = np.zeros((N, K, 3), dtype=np.float64)  # missing -> zeros (low conf)
    clip_ids, frame_ids = [], []
    md_disease, md_exercise, md_camera, md_subject = [], [], [], []
    for i, ip in enumerate(pool_imgs):
        stem = Path(ip).stem
        rec, fidx = _recording_and_frame(stem)
        pr = by_stem.get(stem)
        if pr is not None and pr.keypoints is not None:
            kp = np.asarray(pr.keypoints, dtype=np.float64)
            if kp.shape == (K, 3):
                pred_arr[i] = kp
        clip_ids.append(rec)
        frame_ids.append(fidx)
        m = rec_meta.get(rec, {})
        md_disease.append(m.get("disease"))
        md_exercise.append(m.get("exercise"))
        md_camera.append(m.get("camera_id"))
        md_subject.append(m.get("subject_id"))

    emb_chunks = []
    for s in range(0, N, CHUNK):
        emb_chunks.append(np.asarray(base_model.extract_embeddings(pool_imgs[s:s + CHUNK])))
    embeds = np.concatenate(emb_chunks, axis=0) if emb_chunks else np.zeros((0, 0))
    embeds = np.asarray(embeds, dtype=np.float64)
    if embeds.ndim != 2 or embeds.shape[0] != N:
        _LOG.warning("embeddings shape %s != (%d, D); coreset may degrade",
                     embeds.shape, N)
        embeds = None

    return Pool(
        sample_ids=[Path(p).stem for p in pool_imgs],
        predictions=pred_arr,
        embeddings=embeds,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
        metadata={
            "disease": md_disease,
            "exercise": md_exercise,
            "camera_id": md_camera,
            "subject_id": md_subject,
        },
    )


def _free_cuda() -> None:
    """Release GPU memory between training runs (guarded)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def _train_eval(
    selected_imgs: list[str], out_dir: Path, epochs: int, imgsz: int, seed: int,
    batch: int,
) -> dict:
    ensure_dir(out_dir)
    train_list = out_dir / "train.txt"
    train_list.write_text(
        "\n".join(str(Path(p).resolve()) for p in selected_imgs) + "\n",
        encoding="utf-8",
    )
    data_yaml = _write_round_yaml(out_dir, train_list)
    model = YoloPoseModel(
        {"epochs": epochs, "imgsz": imgsz, "batch": batch, "seed": seed,
         "project": str(out_dir), "name": "train"}
    )
    res = model.train(str(data_yaml), output_dir=str(out_dir / "run"))
    del model
    _free_cuda()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "uncertainty", "coreset", "temporal", "rehabal"])
    ap.add_argument("--budgets", nargs="+", type=float,
                    default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed-frac", type=float, default=0.15,
                    help="fraction of pool used to train the base scoring model")
    ap.add_argument("--out", default=str(WORK))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: random+uncertainty, budgets 0.2/1.0, 1 epoch")
    args = ap.parse_args(argv)

    if args.smoke:
        args.strategies = ["random", "uncertainty"]
        args.budgets = [0.2, 1.0]
        args.epochs = 1
        args.seed_frac = 0.2

    out_root = ensure_dir(Path(args.out))
    pool_imgs = sorted(glob.glob(str(DATASET / "images" / "train" / "*.jpg")))
    if not pool_imgs:
        _LOG.error("No training images under %s; build the dataset first.", DATASET)
        return 1
    N = len(pool_imgs)
    _LOG.info("Pool: %d train images; val = fixed subject-independent split.", N)

    manifest_df = ManifestBuilder.load(MANIFEST)
    rec_meta = _recording_meta(manifest_df)

    # --- base scoring model (trained once on a seed subset) ------------------
    set_seed(args.seeds[0])
    rng = np.random.default_rng(args.seeds[0])
    seed_k = max(1, int(args.seed_frac * N))
    seed_idx = rng.choice(N, size=seed_k, replace=False)
    seed_imgs = [pool_imgs[i] for i in seed_idx]
    _LOG.info("Training base scoring model on %d seed images...", seed_k)
    _train_eval(seed_imgs, out_root / "_base", args.epochs, args.imgsz,
                args.seeds[0], args.batch)
    base_model = YoloPoseModel().load(
        str(out_root / "_base" / "run" / "train" / "weights" / "best.pt")
    )
    pool = _build_pool(pool_imgs, base_model, rec_meta)
    # Free the base scoring model's GPU memory before per-budget training.
    del base_model
    _free_cuda()

    # --- budget curve --------------------------------------------------------
    rows = []
    full_done: dict[float, dict] = {}
    for seed in args.seeds:
        for strategy in args.strategies:
            for budget in args.budgets:
                k = N if budget >= 1.0 else max(1, int(round(budget * N)))
                tag = f"{strategy}_b{budget:g}_s{seed}"
                # budget == 1.0 is strategy-independent (all images) -> compute once
                if budget >= 1.0 and budget in full_done:
                    res = full_done[budget]
                    _LOG.info("[%s] reuse full-budget result", tag)
                else:
                    if budget >= 1.0:
                        sel_idx = list(range(N))
                    else:
                        strat = get_strategy(strategy)
                        set_seed(seed)
                        sel_idx = strat.select(pool, k)
                    sel_imgs = [pool_imgs[i] for i in sel_idx]
                    _LOG.info("[%s] training on %d images...", tag, len(sel_imgs))
                    res = _train_eval(sel_imgs, out_root / tag, args.epochs,
                                      args.imgsz, seed, args.batch)
                    if budget >= 1.0:
                        full_done[budget] = res
                rows.append({
                    "strategy": strategy, "budget": budget, "seed": seed,
                    "n_labeled": k,
                    "pose_map50": res.get("map50"),
                    "pose_map50_95": res.get("map50_95"),
                })
                pd.DataFrame(rows).to_csv(out_root / "curve.csv", index=False)
                _nn = lambda v: float("nan") if v is None else float(v)
                _LOG.info("[%s] pose mAP50=%.4f mAP50-95=%.4f", tag,
                          _nn(res.get("map50")), _nn(res.get("map50_95")))

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "curve.csv", index=False)

    # AUC per strategy (mean over seeds)
    summary = {}
    for strategy in args.strategies:
        sub = df[df.strategy == strategy].groupby("budget")["pose_map50_95"].mean()
        sub = sub.sort_index()
        if len(sub) >= 2:
            auc = float(area_under_learning_curve(list(sub.index), list(sub.values),
                                                  normalize=True))
        else:
            auc = float("nan")
        summary[strategy] = {"auc_map50_95": auc,
                             "points": {float(b): float(v) for b, v in sub.items()}}
    write_json(out_root / "summary.json", summary)
    _LOG.info("AUC (normalized, mAP50-95) per strategy: %s",
              {s: round(v["auc_map50_95"], 4) for s, v in summary.items()})

    # optional plot
    try:
        from rehabal_pose.viz.curves import plot_budget_curve  # noqa
        series = {
            s: (sorted(summary[s]["points"]),
                [summary[s]["points"][b] for b in sorted(summary[s]["points"])])
            for s in args.strategies
        }
        plot_budget_curve(series, out_path=str(out_root / "learning_curve.png"),
                          xlabel="label budget", ylabel="pose mAP50-95")
        _LOG.info("Saved learning_curve.png")
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("plot skipped: %s", exc)

    print("AL_CURVE_DONE", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
