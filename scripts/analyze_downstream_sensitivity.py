"""Recalculate subject weighting and shoulder-proxy sensitivity from saved JSON.

This does not train a model or regenerate predictions. Each --input is NAME=DIR
containing {A,B,ref}_s<seed>_downstream.json. Subject IDs come from --manifest;
the evaluator's documented clip key is <video_id>_rep<index>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ("left_elbow", "right_elbow", "left_knee", "right_knee",
          "left_shoulder", "right_shoulder", "left_hip", "right_hip")
SIX = tuple(j for j in JOINTS if "shoulder" not in j)


def _mean(values):
    array = np.asarray(values, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Sensitivity input must contain finite, nonempty metrics")
    return float(array.mean())


def analyze(inputs, manifest, budget=0.4, seeds=range(6)):
    pairs = pd.read_parquet(manifest)[["video_id", "subject_id"]].drop_duplicates()
    if pairs.isna().any().any() or pairs["video_id"].duplicated().any():
        raise ValueError("Manifest must map every video_id to exactly one subject")
    subjects = {str(v): str(s) for v, s in pairs.itertuples(index=False, name=None)}
    rows = []
    for backbone, directory in inputs:
        for condition in ("A", "B", "ref"):
            for seed in seeds:
                path = Path(directory) / f"{condition}_s{seed}_downstream.json"
                raw = path.read_bytes()
                result = json.loads(raw)
                if result.get("coordinate_frame") != "image-diagonal-normalized":
                    raise ValueError(f"Expected corrected isotropic results: {path}")
                clips = result["budgets"][f"budget_{budget:g}"]["per_clip"]
                grouped = {}
                seen = set()
                for clip in clips:
                    key = clip["clip_id"]
                    match = re.fullmatch(r"(.+)_rep\d+", key)
                    if match is None or key in seen:
                        raise ValueError(f"Invalid or duplicate evaluator clip key: {key}")
                    seen.add(key)
                    subject = subjects[match.group(1)]
                    grouped.setdefault(subject, []).append(clip)
                row = {"backbone": backbone, "condition": condition, "seed": int(seed),
                       "source": str(path.resolve()),
                       "source_sha256": hashlib.sha256(raw).hexdigest(),
                       "subject_clip_counts": {s: len(c) for s, c in sorted(grouped.items())}}
                for kind in ("joint_angle_mae", "rom_error"):
                    for count, joints in ((8, JOINTS), (6, SIX)):
                        per_clip = lambda c: _mean([c[kind][j] for j in joints])
                        row[f"{kind}_{count}_clip_equal"] = _mean([per_clip(c) for c in clips])
                        row[f"{kind}_{count}_subject_equal"] = _mean([
                            _mean([per_clip(c) for c in subset]) for subset in grouped.values()])
                rows.append(row)
    metrics = [f"{kind}_{count}_{weight}" for kind in ("joint_angle_mae", "rom_error")
               for count in (8, 6) for weight in ("clip_equal", "subject_equal")]
    summary = []
    for backbone, _ in inputs:
        for condition in ("A", "B", "ref"):
            subset = [r for r in rows if r["backbone"] == backbone and r["condition"] == condition]
            record = {"backbone": backbone, "condition": condition, "n_seeds": len(subset)}
            record.update({name: _mean([r[name] for r in subset]) for name in metrics})
            summary.append(record)
    return {"budget": budget, "files": len(rows),
            "method": "Angle mean within clip, clip mean within subject, equal subject mean; seed mean last",
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "per_seed": rows, "summary": summary,
            "limitation": "Saved-result sensitivity only; no new participants, training, or inference"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = [value.split("=", 1) for value in args.input]
    if any(len(value) != 2 or not all(value) for value in inputs):
        parser.error("Each --input must be NAME=DIR")
    if len({name for name, _ in inputs}) != len(inputs) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Input names and seeds must be unique")
    result = analyze(inputs, args.manifest, args.budget, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Recalculated {result['files']} saved evaluations: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
