from __future__ import annotations

import argparse
from pathlib import Path

from ..data import ManifestBuilder
from ..utils.config import get, load_yaml, resolve_path
from ..utils.logging import get_logger

_LOG = get_logger(__name__)


def build_manifest(dataset_config: str | Path, out: str | Path) -> Path:
    cfg = load_yaml(dataset_config)
    dataset = get(cfg, "dataset", "unknown")
    root = get(cfg, "paths.root")
    ann_subdir = get(cfg, "paths.annotations_subdir", "labels")

    builder = ManifestBuilder()
    if root:
        ann_dir = resolve_path(Path(root) / ann_subdir)
        if ann_dir.exists():
            for jp in sorted(ann_dir.rglob("*.json")):
                builder.add_row(
                    sample_id=jp.stem,
                    dataset=dataset,
                    annotation_path=str(jp),
                )
        else:
            _LOG.warning("Annotation dir does not exist: %s", ann_dir)
    else:
        _LOG.warning("No paths.root in config; emitting empty manifest.")

    report = builder.validate()
    for issue in report.issues:
        _LOG.warning("validation %s/%s: %s", issue.level, issue.code, issue.message)

    path = builder.save(out)
    _LOG.info("Wrote manifest with %d rows -> %s", len(builder), path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a canonical dataset manifest.")
    parser.add_argument("--dataset-config", required=True, help="datasets/*.yaml")
    parser.add_argument("--out", required=True, help="output .parquet/.csv path")
    args = parser.parse_args(argv)

    build_manifest(args.dataset_config, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
