from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..utils.logging import get_logger
from .pose_base import PoseModel, Prediction

__all__ = ["YoloPoseModel", "is_available"]

_logger = get_logger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "weights": "yolo11n-pose.pt",
    "num_keypoints": 26,
    "device": "auto",
    "imgsz": 640,
    "epochs": 100,
    "batch": 16,
    "project": "outputs/yolo_pose",
    "name": "rehab24_6",
    "seed": 0,
    "conf": 0.25,
    "workers": 0,
}


def is_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("ultralytics") is not None
        and importlib.util.find_spec("torch") is not None
    )


def _require_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "ultralytics is required for YoloPoseModel but could not be "
            "imported. Install it with `pip install ultralytics` (and a working "
            "torch build)."
        ) from exc
    return YOLO


class YoloPoseModel(PoseModel):
    name = "yolo-pose"

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config: dict[str, Any] = dict(_DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.num_keypoints: int = int(self.config["num_keypoints"])
        self._model: Any = None
        self.weights_path: Optional[str] = None

    def _resolve_device(self, override: Optional[str] = None) -> str:
        dev = override or self.config.get("device", "auto")
        if dev != "auto":
            return str(dev)
        try:
            import torch  # type: ignore

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch missing
            return "cpu"

    def _ensure_model(self, weights: Optional[str] = None) -> Any:
        if weights is not None:
            self.load(weights)
        if self._model is None:
            self.load(self.config["weights"])
        return self._model

    def load(self, weights: str | Path) -> "YoloPoseModel":
        YOLO = _require_yolo()
        self._model = YOLO(str(weights))
        self.weights_path = str(weights)
        _logger.info("Loaded YOLO-Pose weights: %s", weights)
        return self

    def train(
        self,
        data_yaml: str | Path,
        output_dir: str | Path | None = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        cfg = dict(self.config)
        if config:
            cfg.update(config)

        project = str(output_dir) if output_dir is not None else cfg["project"]
        device = self._resolve_device(cfg.get("device"))
        model = self._ensure_model(cfg.get("weights"))

        reserved = {
            "weights", "num_keypoints", "device", "project", "name",
            "epochs", "imgsz", "batch", "seed", "conf", "workers",
        }
        extra = {k: v for k, v in cfg.items() if k not in reserved}

        _logger.info(
            "Training YOLO-Pose: data=%s epochs=%s imgsz=%s batch=%s device=%s",
            data_yaml, cfg["epochs"], cfg["imgsz"], cfg["batch"], device,
        )
        results = model.train(
            data=str(data_yaml),
            epochs=int(cfg["epochs"]),
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            device=device,
            project=str(project),
            name=str(cfg["name"]),
            seed=int(cfg.get("seed", 0)),
            workers=int(cfg.get("workers", 0)),
            deterministic=True,
            **extra,
        )

        save_dir = Path(getattr(results, "save_dir", Path(project) / cfg["name"]))
        best = save_dir / "weights" / "best.pt"
        last = save_dir / "weights" / "last.pt"
        if best.exists():
            self.load(best)

        map50_95, map50, raw = _extract_pose_metrics(results)
        return {
            "weights": str(best) if best.exists() else None,
            "last_weights": str(last) if last.exists() else None,
            "save_dir": str(save_dir),
            "map50_95": map50_95,
            "map50": map50,
            "metrics": raw,
            "epochs": int(cfg["epochs"]),
            "device": device,
        }

    def predict(
        self,
        inputs: str | Path | Sequence[str | Path],
        output_path: str | Path | None = None,
        config: Optional[dict[str, Any]] = None,
    ) -> list[Prediction]:
        cfg = dict(self.config)
        if config:
            cfg.update(config)
        device = self._resolve_device(cfg.get("device"))
        model = self._ensure_model((config or {}).get("weights"))

        source = _normalize_sources(inputs)
        kwargs: dict[str, Any] = {
            "conf": float(cfg.get("conf", 0.25)),
            "imgsz": int(cfg["imgsz"]),
            "device": device,
            "verbose": False,
            "workers": int(cfg.get("workers", 0)),
        }
        if output_path is not None:
            kwargs["save"] = True
            kwargs["project"] = str(output_path)

        results = model.predict(source=source, **kwargs)

        preds: list[Prediction] = []
        for res in results:
            sample_id = Path(getattr(res, "path", "frame")).stem
            kpts, bbox, score = _best_instance(res, self.num_keypoints)
            if kpts is None:
                continue
            preds.append(
                Prediction(
                    sample_id=sample_id,
                    keypoints=kpts,
                    bbox=bbox,
                    score=score,
                )
            )
        return preds

    def evaluate(
        self,
        data_yaml: str | Path,
        output_dir: str | Path | None = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        cfg = dict(self.config)
        if config:
            cfg.update(config)
        device = self._resolve_device(cfg.get("device"))
        model = self._ensure_model((config or {}).get("weights"))

        kwargs: dict[str, Any] = {
            "data": str(data_yaml),
            "imgsz": int(cfg["imgsz"]),
            "device": device,
            "verbose": False,
            "workers": int(cfg.get("workers", 0)),
        }
        if output_dir is not None:
            kwargs["project"] = str(output_dir)
            kwargs["name"] = str(cfg["name"])

        results = model.val(**kwargs)
        map50_95, map50, raw = _extract_pose_metrics(results)
        return {
            "map50_95": map50_95,
            "map50": map50,
            "metrics": raw,
            "save_dir": str(getattr(results, "save_dir", "")),
            "device": device,
        }

    def extract_embeddings(
        self,
        inputs: str | Path | Sequence[str | Path],
        output_path: str | Path | None = None,
        config: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        cfg = dict(self.config)
        if config:
            cfg.update(config)
        device = self._resolve_device(cfg.get("device"))
        model = self._ensure_model((config or {}).get("weights"))
        source = _normalize_sources(inputs)

        feats = model.embed(
            source=source,
            imgsz=int(cfg["imgsz"]),
            device=device,
            verbose=False,
            workers=int(cfg.get("workers", 0)),
        )

        vecs: list[np.ndarray] = []
        for f in feats:
            arr = _to_numpy(f).reshape(-1)
            vecs.append(arr.astype(np.float32))
        if not vecs:
            return np.zeros((0, 0), dtype=np.float32)
        out = np.stack(vecs, axis=0)

        if output_path is not None:
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            np.save(outp, out)
            _logger.info("Saved embeddings %s -> %s", out.shape, outp)
        return out


def _normalize_sources(
    inputs: str | Path | Sequence[str | Path],
) -> Any:
    if isinstance(inputs, (str, Path)):
        return str(inputs)
    return [str(p) for p in inputs]


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if detach is not None:
        x = detach()
        cpu = getattr(x, "cpu", None)
        if cpu is not None:
            x = cpu()
        npy = getattr(x, "numpy", None)
        if npy is not None:
            return np.asarray(npy())
    return np.asarray(x)


def _best_instance(
    res: Any, num_keypoints: int
) -> tuple[Optional[np.ndarray], Optional[list[float]], Optional[float]]:
    kpts_obj = getattr(res, "keypoints", None)
    boxes = getattr(res, "boxes", None)
    if kpts_obj is None:
        return None, None, None

    data = getattr(kpts_obj, "data", None)
    if data is None:
        return None, None, None
    kp = _to_numpy(data)
    if kp.ndim != 3 or kp.shape[0] == 0:
        return None, None, None

    best = 0
    score: Optional[float] = None
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        conf = _to_numpy(boxes.conf).reshape(-1)
        if conf.size:
            best = int(np.argmax(conf))
            score = float(conf[best])

    inst = kp[best]
    if inst.shape[1] == 2:
        conf_col = np.ones((inst.shape[0], 1), dtype=inst.dtype)
        inst = np.concatenate([inst, conf_col], axis=1)
    kp33 = inst[:, :3].astype(np.float64)

    bbox: Optional[list[float]] = None
    if boxes is not None and getattr(boxes, "xywh", None) is not None:
        xywh = _to_numpy(boxes.xywh)
        if xywh.ndim == 2 and xywh.shape[0] > best:
            cx, cy, w, h = (float(v) for v in xywh[best][:4])
            bbox = [cx - w / 2.0, cy - h / 2.0, w, h]

    return kp33, bbox, score


def _extract_pose_metrics(
    results: Any,
) -> tuple[Optional[float], Optional[float], dict[str, Any]]:
    map50_95: Optional[float] = None
    map50: Optional[float] = None
    raw: dict[str, Any] = {}

    pose = getattr(results, "pose", None)
    if pose is not None:
        if getattr(pose, "map", None) is not None:
            try:
                map50_95 = float(pose.map)
            except Exception:  # pragma: no cover
                pass
        if getattr(pose, "map50", None) is not None:
            try:
                map50 = float(pose.map50)
            except Exception:  # pragma: no cover
                pass

    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        for k, v in rd.items():
            try:
                raw[str(k)] = float(v)
            except Exception:  # pragma: no cover
                raw[str(k)] = v
        if map50_95 is None:
            for key in ("metrics/mAP50-95(P)", "metrics/mAP50-95(pose)"):
                if key in raw:
                    map50_95 = raw[key]
                    break
        if map50 is None:
            for key in ("metrics/mAP50(P)", "metrics/mAP50(pose)"):
                if key in raw:
                    map50 = raw[key]
                    break

    return map50_95, map50, raw
