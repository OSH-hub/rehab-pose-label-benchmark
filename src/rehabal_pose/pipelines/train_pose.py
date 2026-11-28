from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ..utils.config import load_yaml
from ..utils.io import write_json
from ..utils.logging import get_logger

__all__ = ["train_pose", "load_model_config", "main"]

_LOG = get_logger(__name__)


def load_model_config(
    model_config: str | Path | None,
    *,
    profile: Optional[str] = None,
) -> dict[str, Any]:
    if model_config is None:
        return {}
    cfg = load_yaml(model_config)
    base = cfg.get("model", {})
    out: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    if profile:
        overlay = cfg.get(profile)
        if not isinstance(overlay, dict):
            raise KeyError(
                f"profile {profile!r} not found in {model_config} "
                f"(available top-level keys: {sorted(cfg)})"
            )
        out.update(overlay)
    return out


def train_pose(
    data_yaml: str | Path,
    output_dir: str | Path | None = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from ..models.yolo_pose import YoloPoseModel

    cfg = dict(config or {})
    model = YoloPoseModel(cfg)
    _LOG.info(
        "train_pose: data=%s output_dir=%s epochs=%s imgsz=%s batch=%s",
        data_yaml, output_dir, cfg.get("epochs"), cfg.get("imgsz"), cfg.get("batch"),
    )
    result = model.train(data_yaml, output_dir=output_dir, config=cfg)
    return result


def _write_metrics(
    result: dict[str, Any],
    output_dir: str | Path | None,
) -> Path:
    target = output_dir or result.get("save_dir") or "."
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    write_json(metrics_path, result)
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO-Pose 2D pose model on a YOLO-pose data.yaml."
    )
    parser.add_argument(
        "--data", required=True, help="ultralytics pose data.yaml"
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="configs/models/yolo_pose.yaml (omit to use built-in defaults)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="config overlay section (e.g. 'smoke') applied over model:",
    )
    parser.add_argument(
        "--output-dir", default=None, help="run output root (ultralytics project)"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="'auto', 'cpu', or CUDA index like '0'"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_model_config(args.model_config, profile=args.profile)

    for key, value in (
        ("epochs", args.epochs),
        ("imgsz", args.imgsz),
        ("batch", args.batch),
        ("device", args.device),
        ("seed", args.seed),
    ):
        if value is not None:
            config[key] = value

    result = train_pose(args.data, args.output_dir, config)

    metrics_path = _write_metrics(result, args.output_dir)
    weights = result.get("weights")
    _LOG.info("Wrote training metrics -> %s", metrics_path)
    _LOG.info(
        "Best weights: %s (mAP50-95=%s, mAP50=%s)",
        weights, result.get("map50_95"), result.get("map50"),
    )
    print(f"metrics: {metrics_path}")
    print(f"weights: {weights}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
