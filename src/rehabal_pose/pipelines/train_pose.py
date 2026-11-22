"""Train a YOLO-Pose 2D pose model on a YOLO-pose dataset (REHAB24-6).

It loads a model config, constructs a
:class:`rehabal_pose.models.yolo_pose.YoloPoseModel`, fine-tunes it on an
ultralytics pose ``data.yaml`` (built by the frame extractor +
:func:`rehabal_pose.data.yolo_dataset.write_pose_data_yaml`), writes a
``metrics.json`` next to the run, and echoes the best-weights path.

Import-safety requirement: no heavy imports at module load. ``torch`` / ``ultralytics`` /
``cv2`` are never imported when this module is imported: ``YoloPoseModel`` is
imported lazily inside :func:`train_pose`, and even that module keeps the heavy
stack behind its method bodies. The default test suite therefore imports this
pipeline without dragging in the ML stack.

Usage::

    python -m rehabal_pose.pipelines.train_pose \
        --data outputs/rehab24_6_yolo/data.yaml \
        --model-config configs/models/yolo_pose.yaml \
        --output-dir outputs/yolo_pose

    # fast pipeline check (1 epoch, imgsz 320, batch 2):
    python -m rehabal_pose.pipelines.train_pose \
        --data outputs/rehab24_6_yolo/data.yaml \
        --model-config configs/models/yolo_pose.yaml \
        --profile smoke
"""

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
    """Load a YOLO-Pose model config into a flat ``YoloPoseModel`` override dict.

    The YAML is expected to hold a ``model:`` mapping whose keys map 1:1 onto
    :class:`~rehabal_pose.models.yolo_pose.YoloPoseModel` config keys (``weights``,
    ``num_keypoints``, ``imgsz``, ``epochs``, ``batch``, ``device``, ``patience``,
    ...). When *profile* is given, the YAML's ``<profile>:`` mapping is overlaid
    on top of ``model:`` (shallow merge), so e.g. ``profile="smoke"`` reduces
    epochs/imgsz/batch for a fast verification run.

    Parameters
    ----------
    model_config:
        Path to ``configs/models/yolo_pose.yaml`` (or ``None`` -> empty dict, so
        the model falls back to its built-in defaults).
    profile:
        Optional overlay section name (e.g. ``"smoke"``).

    Returns
    -------
    dict
        Flat override dict suitable for ``YoloPoseModel(config)`` /
        ``YoloPoseModel.train(..., config=config)``.
    """
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
    """Fine-tune a YOLO-Pose model and return its training-result dict.

    This is the programmatic entrypoint the active-learning loop wraps as its
    injected ``train_fn``. It constructs a
    :class:`~rehabal_pose.models.yolo_pose.YoloPoseModel` from *config* and calls
    :meth:`~rehabal_pose.models.yolo_pose.YoloPoseModel.train`.

    Parameters
    ----------
    data_yaml:
        Path to an ultralytics pose ``data.yaml``.
    output_dir:
        Run output root (ultralytics ``project``). Falls back to the config's
        ``project`` when ``None``.
    config:
        Flat ``YoloPoseModel`` override dict (see :func:`load_model_config`).

    Returns
    -------
    dict
        The ``YoloPoseModel.train`` result: ``{"weights", "last_weights",
        "save_dir", "map50_95", "map50", "metrics", "epochs", "device"}``.
    """
    # Lazy import: yolo_pose is heavy-import-free at module load, but importing
    # it here keeps the pipeline module's import graph minimal and makes the
    # "no heavy deps" contract obvious.
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
    """Write *result* as ``metrics.json`` and return its path.

    Destination preference: ``output_dir`` if given, else the run's
    ``save_dir`` from *result*, else the current directory.
    """
    target = output_dir or result.get("save_dir") or "."
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    write_json(metrics_path, result)
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse args, train, write ``metrics.json``, echo weights."""
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

    # CLI flags override the config file (only when explicitly provided).
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
