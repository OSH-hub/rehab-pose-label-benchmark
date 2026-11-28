from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..al import Pool
from ..utils.logging import get_logger

__all__ = ["infer_pool", "save_pool", "load_pool", "items_from_manifest", "main"]

_LOG = get_logger(__name__)

_META_KEYS = ("disease", "exercise", "camera_id", "standardtype")


def _is_dataframe(obj: Any) -> bool:
    return (
        hasattr(obj, "iterrows")
        and hasattr(obj, "columns")
        and hasattr(obj, "to_dict")
    )


def _clean(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _normalize_items(image_items: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if _is_dataframe(image_items):
        cols = set(image_items.columns)
        for _, row in image_items.iterrows():
            rec: dict[str, Any] = {}
            img = row.get("image_path") if "image_path" in cols else None
            img = _clean(img)
            if img is None and "video_path" in cols:
                img = _clean(row.get("video_path"))
            if img is None:
                raise ValueError(
                    "manifest row has neither image_path nor video_path; "
                    "extract frames first or pass --images-dir"
                )
            rec["image_path"] = str(img)
            sid = _clean(row.get("sample_id")) if "sample_id" in cols else None
            rec["sample_id"] = str(sid) if sid is not None else Path(str(img)).stem
            for key in ("video_id", "frame_id", *_META_KEYS):
                if key in cols:
                    rec[key] = _clean(row.get(key))
            records.append(rec)
        return records

    if isinstance(image_items, (str, Path)):
        image_items = [image_items]

    for item in image_items:
        if isinstance(item, (str, Path)):
            p = str(item)
            records.append({"image_path": p, "sample_id": Path(p).stem})
        elif isinstance(item, dict):
            rec = dict(item)
            img = rec.get("image_path")
            if img is None:
                raise ValueError(f"item dict missing 'image_path': {item!r}")
            rec["image_path"] = str(img)
            rec.setdefault("sample_id", Path(rec["image_path"]).stem)
            rec["sample_id"] = str(rec["sample_id"])
            records.append(rec)
        else:
            raise TypeError(
                f"Unsupported image item type {type(item).__name__}; expected "
                "str / Path / dict"
            )
    return records


def items_from_manifest(
    manifest: Any,
    images_dir: str | Path | None = None,
    *,
    image_ext: str = ".jpg",
) -> list[dict[str, Any]]:
    records = _normalize_items(manifest)
    if images_dir is not None:
        base = Path(images_dir)
        for rec in records:
            ip = rec.get("image_path", "")
            if not ip or not Path(ip).suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp",
            } or not Path(ip).is_file():
                rec["image_path"] = str(base / f"{rec['sample_id']}{image_ext}")
    return records


def infer_pool(
    model: Any,
    image_items: Any,
    output_path: str | Path | None = None,
    config: Optional[dict[str, Any]] = None,
) -> Pool:
    cfg = dict(config or {})
    records = _normalize_items(image_items)
    n = len(records)

    sample_ids = [r["sample_id"] for r in records]
    image_paths = [r["image_path"] for r in records]

    preds: Sequence[Any] = []
    if n:
        predict_kwargs = cfg.get("predict_config")
        if predict_kwargs is not None:
            preds = model.predict(image_paths, config=predict_kwargs)
        else:
            preds = model.predict(image_paths)

    by_stem: dict[str, Any] = {}
    for p in preds:
        stem = str(getattr(p, "sample_id", ""))
        prev = by_stem.get(stem)
        if prev is None:
            by_stem[stem] = p
        else:
            ps = getattr(p, "score", None) or -np.inf
            qs = getattr(prev, "score", None) or -np.inf
            if ps > qs:
                by_stem[stem] = p

    num_kp = cfg.get("num_keypoints")
    if num_kp is None:
        num_kp = getattr(model, "num_keypoints", None)
    if num_kp is None:
        for p in by_stem.values():
            num_kp = int(np.asarray(p.keypoints).shape[0])
            break
    num_kp = int(num_kp or 26)

    predictions = np.zeros((n, num_kp, 3), dtype=np.float64)
    det_scores: list[Optional[float]] = [None] * n
    for i, rec in enumerate(records):
        stem = Path(rec["image_path"]).stem
        p = by_stem.get(stem)
        if p is None:
            continue
        kp = np.asarray(p.keypoints, dtype=np.float64)
        k = min(kp.shape[0], num_kp)
        predictions[i, :k, :] = kp[:k, :3]
        sc = getattr(p, "score", None)
        det_scores[i] = float(sc) if sc is not None else None

    n_detected = sum(1 for s in det_scores if s is not None)
    _LOG.info(
        "infer_pool: %d/%d images produced a detection (K=%d).",
        n_detected, n, num_kp,
    )

    has_video = any(r.get("video_id") is not None for r in records)
    clip_ids = [r.get("video_id") for r in records] if has_video else None

    has_frame = any(r.get("frame_id") is not None for r in records)
    frame_ids: Optional[list[int]] = None
    if has_frame:
        frame_ids = []
        for r in records:
            fid = r.get("frame_id")
            frame_ids.append(int(fid) if fid is not None else 0)

    metadata: dict[str, list[Any]] = {}
    for key in _META_KEYS:
        if any(r.get(key) is not None for r in records):
            metadata[key] = [r.get(key) for r in records]
    if n_detected:
        metadata["det_score"] = list(det_scores)

    embeddings: Optional[np.ndarray] = None
    if cfg.get("with_embeddings") and n:
        embed_kwargs = cfg.get("embed_config")
        if embed_kwargs is not None:
            emb = model.extract_embeddings(image_paths, config=embed_kwargs)
        else:
            emb = model.extract_embeddings(image_paths)
        emb = np.asarray(emb, dtype=np.float64)
        if emb.ndim == 2 and emb.shape[0] == n:
            embeddings = emb
        else:
            _LOG.warning(
                "extract_embeddings returned shape %s for %d samples; "
                "dropping embeddings.", emb.shape, n,
            )

    pool = Pool(
        sample_ids=sample_ids,
        predictions=predictions,
        embeddings=embeddings,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
        metadata=metadata or None,
    )

    if output_path is not None:
        save_pool(pool, output_path)

    return pool


def save_pool(pool: Pool, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        return _save_pool_npz(pool, p)
    if suffix == ".parquet":
        return _save_pool_parquet(pool, p)
    raise ValueError(f"Unsupported pool extension {p.suffix!r}; use .npz or .parquet")


def load_pool(path: str | Path) -> Pool:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Pool file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".npz":
        return _load_pool_npz(p)
    if suffix == ".parquet":
        return _load_pool_parquet(p)
    raise ValueError(f"Unsupported pool extension {p.suffix!r}; use .npz or .parquet")


def _save_pool_npz(pool: Pool, p: Path) -> Path:
    aux = {
        "sample_ids": list(pool.sample_ids),
        "clip_ids": list(pool.clip_ids) if pool.clip_ids is not None else None,
        "frame_ids": (
            [int(f) for f in pool.frame_ids] if pool.frame_ids is not None else None
        ),
        "metadata": (
            {k: list(v) for k, v in pool.metadata.items()}
            if pool.metadata is not None
            else None
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "aux_json": np.asarray(json.dumps(aux, ensure_ascii=False)),
    }
    if pool.predictions is not None:
        arrays["predictions"] = np.asarray(pool.predictions, dtype=np.float64)
    if pool.embeddings is not None:
        arrays["embeddings"] = np.asarray(pool.embeddings, dtype=np.float64)
    np.savez(p, **arrays)
    _LOG.info("Saved pool (%d samples) -> %s", pool.size, p)
    return p


def _load_pool_npz(p: Path) -> Pool:
    with np.load(p, allow_pickle=False) as d:
        aux = json.loads(str(d["aux_json"]))
        predictions = d["predictions"] if "predictions" in d.files else None
        embeddings = d["embeddings"] if "embeddings" in d.files else None
    return Pool(
        sample_ids=list(aux["sample_ids"]),
        predictions=predictions,
        embeddings=embeddings,
        clip_ids=aux.get("clip_ids"),
        frame_ids=aux.get("frame_ids"),
        metadata=aux.get("metadata"),
    )


def _save_pool_parquet(pool: Pool, p: Path) -> Path:
    import pandas as pd

    n = pool.size
    data: dict[str, Any] = {"sample_id": list(pool.sample_ids)}
    if pool.clip_ids is not None:
        data["clip_id"] = list(pool.clip_ids)
    if pool.frame_ids is not None:
        data["frame_id"] = list(pool.frame_ids)
    if pool.metadata is not None:
        for k, v in pool.metadata.items():
            data[f"meta_{k}"] = list(v)
    if pool.predictions is not None:
        preds = np.asarray(pool.predictions, dtype=np.float64)
        k = preds.shape[1]
        for j in range(k):
            data[f"kp{j}_x"] = preds[:, j, 0]
            data[f"kp{j}_y"] = preds[:, j, 1]
            data[f"kp{j}_c"] = preds[:, j, 2]
    if pool.embeddings is not None:
        emb = np.asarray(pool.embeddings, dtype=np.float64)
        for j in range(emb.shape[1]):
            data[f"emb{j}"] = emb[:, j]
    df = pd.DataFrame(data, index=range(n))
    df.to_parquet(p, engine="pyarrow", index=False)
    _LOG.info("Saved pool (%d samples) -> %s", n, p)
    return p


def _load_pool_parquet(p: Path) -> Pool:
    import pandas as pd

    df = pd.read_parquet(p, engine="pyarrow")
    cols = list(df.columns)
    sample_ids = [str(s) for s in df["sample_id"].tolist()]

    kp_idx = sorted(
        int(c[2:-2]) for c in cols if c.startswith("kp") and c.endswith("_x")
    )
    predictions = None
    if kp_idx:
        k = max(kp_idx) + 1
        n = len(df)
        predictions = np.zeros((n, k, 3), dtype=np.float64)
        for j in kp_idx:
            predictions[:, j, 0] = df[f"kp{j}_x"].to_numpy()
            predictions[:, j, 1] = df[f"kp{j}_y"].to_numpy()
            predictions[:, j, 2] = df[f"kp{j}_c"].to_numpy()

    emb_idx = sorted(int(c[3:]) for c in cols if c.startswith("emb"))
    embeddings = None
    if emb_idx:
        embeddings = np.stack(
            [df[f"emb{j}"].to_numpy() for j in emb_idx], axis=1
        ).astype(np.float64)

    clip_ids = df["clip_id"].tolist() if "clip_id" in cols else None
    frame_ids = (
        [int(f) for f in df["frame_id"].tolist()] if "frame_id" in cols else None
    )
    metadata: dict[str, list[Any]] = {}
    for c in cols:
        if c.startswith("meta_"):
            metadata[c[len("meta_"):]] = df[c].tolist()

    return Pool(
        sample_ids=sample_ids,
        predictions=predictions,
        embeddings=embeddings,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
        metadata=metadata or None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run pose inference over the unlabeled pool -> AL Pool."
    )
    parser.add_argument("--weights", required=True, help="trained YOLO-pose .pt")
    parser.add_argument("--manifest", required=True, help="pool manifest (.parquet/.csv)")
    parser.add_argument(
        "--images-dir",
        help="root holding extracted per-sample frames (<sample_id>.jpg)",
    )
    parser.add_argument("--out", required=True, help="output pool (.npz or .parquet)")
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="also compute backbone embeddings (coreset/rehabal strategies)",
    )
    parser.add_argument("--num-keypoints", type=int, default=26)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    from ..data import ManifestBuilder
    from ..models.yolo_pose import YoloPoseModel

    df = ManifestBuilder.load(args.manifest)
    items = items_from_manifest(df, args.images_dir)

    model = YoloPoseModel(
        {
            "weights": args.weights,
            "num_keypoints": args.num_keypoints,
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
        }
    )
    model.load(args.weights)

    pool = infer_pool(
        model,
        items,
        output_path=args.out,
        config={
            "with_embeddings": bool(args.with_embeddings),
            "num_keypoints": args.num_keypoints,
        },
    )
    _LOG.info("infer_pool wrote %d samples to %s", pool.size, args.out)
    print(json.dumps({"n": pool.size, "out": str(args.out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
