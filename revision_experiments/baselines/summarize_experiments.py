"""Summarize the matched YOLO11n follow-up suite without inventing missing results."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import unittest
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

METHODS = ["A", "B", "ref", "A_jointmask", "A_flipcons"]
LABELS = {
    "A": "A0: original self-training",
    "B": "Boracle: GT correction + rejection",
    "ref": "ref: full-GT comparator",
    "A_jointmask": "Joint-confidence masking",
    "A_flipcons": "Flip-consensus pseudo-labels",
}
SCHEMA_VERSION = 1


def lp(value):
    s = str(Path(value).absolute())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        if s.startswith("\\\\"):
            return "\\\\?\\UNC\\" + s[2:]
        return "\\\\?\\" + s
    return s


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with open(lp(path), encoding="utf-8-sig") as f:
        return json.load(f)


def read_csv(path):
    with open(lp(path), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with open(lp(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with open(lp(path), "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_text(path, value):
    with open(lp(path), "x", encoding="utf-8") as f:
        f.write(value)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(lp(path), "x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def number(value, name):
    try:
        x = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name}: invalid number {value!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{name}: nonfinite number")
    return x


def integer(value, name):
    x = number(value, name)
    if x != int(x):
        raise ValueError(f"{name}: integer required")
    return int(x)


def proportion(value, name):
    x = number(value, name)
    if not 0 <= x <= 1:
        raise ValueError(f"{name}: expected a fraction in [0,1], got {x}")
    return x


def key(row):
    method = row.get("condition", row.get("method"))
    if method is None:
        raise ValueError("Missing condition/method")
    return str(method), integer(row["seed"], "seed"), round(number(row["budget"], "budget"), 8)


def expected_keys(protocol, external=False):
    budgets = protocol["penn_budgets"] if external else protocol["budgets"]
    return set(itertools.product(protocol["methods"], protocol["seeds"], budgets))


def normalized_budget_auc(budgets, values, planned):
    b = np.asarray(budgets, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(b) != len(set(b)) or set(b) != set(planned):
        raise ValueError("AUC requires exactly one observation at every planned budget")
    if b.shape != y.shape or not np.all(np.isfinite(y)):
        raise ValueError("Invalid AUC inputs")
    ix = np.argsort(b)
    b, y = b[ix], y[ix]
    if len(b) < 2 or b[-1] <= b[0]:
        raise ValueError("AUC needs at least two distinct budgets")
    area = np.sum((b[1:] - b[:-1]) * (y[1:] + y[:-1]) / 2)
    return float(area / (b[-1] - b[0]))


def paired_clip_bootstrap(a, b=None, n_boot=10000, seed=20261012):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Bootstrap needs finite, nonempty clip values")
    if b is not None:
        b = np.asarray(b, dtype=float)
        if b.shape != a.shape or not np.all(np.isfinite(b)):
            raise ValueError("Paired clip arrays must have the same finite shape")
        a = a - b
    mean = float(a.mean())
    if len(a) < 2:
        return {"estimate": mean, "ci_low": None, "ci_high": None, "n_clips": len(a)}
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, n_boot, 128):
        n = min(128, n_boot - start)
        ix = rng.integers(0, len(a), size=(n, len(a)))
        values.extend(a[ix].mean(axis=1))
    lo, hi = np.quantile(values, [.025, .975])
    return {"estimate": mean, "ci_low": float(lo), "ci_high": float(hi), "n_clips": len(a)}


def protocol_template(args):
    return {
        "schema_version": SCHEMA_VERSION,
        "locked_at_utc": now(),
        "purpose": "Prospectively fixed exploratory extension; not preregistered original study",
        "methods": METHODS,
        "display_names": LABELS,
        "seeds": [0, 1, 2, 3, 4, 5],
        "budgets": [.05, .10, .20, .40],
        "penn_budgets": args.penn_budgets,
        "backbone": "yolo11n-pose",
        "condition_provenance": {"A": "original_reused", "B": "original_reused", "ref": "original_reused", "A_jointmask": "new_followup", "A_flipcons": "new_followup"},
        "epochs_per_round": 30,
        "batch": 16,
        "train_imgsz": 416,
        "external_imgsz": 640,
        "detector_confidence": .25,
        "person_selection": "highest detection confidence, without GT association",
        "source_train_subjects": list(range(1, 9)),
        "source_validation_subjects": [9, 10],
        "source_pool_size_expected": args.pool_size,
        "source_validation_frames_reported_original": 989,
        "source_validation_subject9_frames_reported_original": 934,
        "source_validation_subject10_frames_reported_original": 55,
        "source_validation_role": "checkpoint/hyperparameter development; internal validation",
        "external_test_role": "Penn Action official video test split; no adaptation or tuning",
        "penn_test_clips_expected": 1068,
        "penn_test_frames_expected": 76298,
        "primary_external_metric": "clip-macro PCK@0.20, 12 common limb joints",
        "secondary_external_metric": "micro PCK@0.20 and clip-macro PCK@0.10",
        "pck_normalizer": "max(GT bbox width, GT bbox height)",
        "missing_detection_rule": "all eligible visible GT joints count as incorrect",
        "missing_prediction_rule": "do not drop eligible GT joints from denominator",
        "invalid_gt_rule": "exclude only prespecified invalid GT, with exclusion counts",
        "bootstrap_unit": "paired clip, all frames/joints retained as a cluster",
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": 20261012,
        "training_seed_summary": "all per-seed values plus mean and population SD (ddof=0)",
        "primary_contrast": ["B", "A_flipcons"],
        "other_contrasts": [["B", "A"], ["B", "A_jointmask"], ["B", "ref"]],
        "multiplicity": "exploratory effect estimates; no significance labels or p-value claims",
        "budget_auc": "trapezoidal source mAP50_95 vs accepted-frame fraction / (0.40-0.05)",
        "external_threshold_auc": "not a label-budget AUC; not computed by this script",
        "interpretation": [
            "B-minus-A is the total adaptive correction/rejection policy contrast",
            "Acceptance counts are not matched human annotation costs",
            "GT-based B is an oracle; joint-mask/flip-consensus are custom baselines",
            "Original A/B/ref checkpoints/results are reused; the two custom arms are new follow-up runs",
            "Pooling old/new observations requires verified split, initial GT set, backbone and training environment comparability",
            "New source splits cannot undo prior development exposure of all ten REHAB subjects",
            "Penn video-cluster intervals are conditional on fitted checkpoints, not patient-population or training-seed intervals",
            "Penn Action is a sports/action dataset, not a rehabilitation patient cohort",
        ],
    }


def check_protocol(p):
    if p.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported analysis protocol version")
    for name in ("methods", "seeds", "budgets", "penn_budgets"):
        if not p.get(name) or len(p[name]) != len(set(p[name])):
            raise ValueError(f"Protocol {name} must be nonempty and unique")
    if set(p.get("condition_provenance", {})) != set(p["methods"]):
        raise ValueError("Protocol needs explicit original/new provenance for every method")
    if set(p["methods"]) != set(METHODS):
        raise ValueError("This analysis expects the five explicitly planned methods")
    if set(p["seeds"]) != {0, 1, 2, 3, 4, 5}:
        raise ValueError("This protocol is the matched six-seed follow-up suite")
    if p["budgets"] != [.05, .10, .20, .40]:
        raise ValueError("Budget changes require an explicitly revised analysis")
    if not set(p["penn_budgets"]).issubset(p["budgets"]):
        raise ValueError("External budgets must be a subset of trained budgets")
    if integer(p["bootstrap_resamples"], "bootstrap_resamples") < 1000:
        raise ValueError("Use at least 1000 cluster-bootstrap draws")


def load_training(base, protocol, additional_csvs=(), additional_audits=()):
    csv_paths = []
    if base is not None:
        master = base / "clinician_curve.csv"
        if Path(lp(master)).is_file():
            csv_paths.append(master)
        elif Path(lp(base)).is_dir():
            csv_paths.extend(sorted(Path(lp(base)).glob("*/experiment_curve.csv")))
    csv_paths.extend(additional_csvs)
    csv_paths = list(dict.fromkeys(str(q) for q in csv_paths))
    audit_paths = list(additional_audits)
    if base is not None and Path(lp(base)).is_dir():
        audit_paths.extend(sorted(Path(lp(base)).rglob("audit_rounds.jsonl")))
    audit_paths = list(dict.fromkeys(str(q) for q in audit_paths))
    if not csv_paths:
        return {}, [], [{"type": "missing_training_inputs", "path": str(base)}]
    audits, issues = {}, []
    sources = [Path(q) for q in csv_paths + audit_paths]
    for audit in audit_paths:
        with open(lp(audit), encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get("status") == "complete":
                    k = key(r)
                    if k in audits and audits[k].get("checkpoint_sha256") != r.get("checkpoint_sha256"):
                        raise ValueError(f"Conflicting completed audits for {k}")
                    audits[k] = r
    rows = {}
    for csv_path in csv_paths:
        for r in read_csv(csv_path):
            if r.get("status", "complete") != "complete":
                issues.append({"type":"training_csv_row_not_complete", "source":str(csv_path), "condition":r.get("condition"), "seed":r.get("seed"), "budget":r.get("budget"), "status":r.get("status")})
                continue
            condition = str(r.get("condition", r.get("method")))
            sigma = str(r.get("sigma", "")).strip().lower()
            if condition == "B" and sigma not in ("", "nan", "none", "null"):
                if number(sigma, "sigma") != 0:
                    continue
            k = key(r)
            if k not in expected_keys(protocol):
                raise ValueError(f"Unexpected training key {k}; do not pool unrelated studies")
            if k in rows:
                raise ValueError(f"Duplicate training key {k}")
            item = {
                "condition": k[0], "seed": k[1], "budget": k[2],
                "provenance": protocol["condition_provenance"][k[0]],
                "source_csv": str(csv_path),
                "n_labeled": integer(r["n_labeled"], "n_labeled"),
                "map50_95": proportion(r["map50_95"], "map50_95"),
                "map50": proportion(r["map50"], "map50"),
            }
            wanted = round(protocol["source_pool_size_expected"] * k[2])
            if item["n_labeled"] != wanted:
                raise ValueError(f"{k}: n_labeled={item['n_labeled']} differs from locked pool/budget target {wanted}")
            audit = audits.get(k)
            if audit is None:
                issues.append({"type": "training_completion_not_verified", "key": list(k)})
                continue
            sha = audit.get("checkpoint_sha256", "")
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
                raise ValueError(f"{k}: completed audit lacks valid checkpoint SHA256")
            explicit_provenance = audit.get("provenance")
            if explicit_provenance is not None and explicit_provenance != item["provenance"]:
                raise ValueError(f"Unexpected provenance for {k}: {explicit_provenance}")
            item["checkpoint_sha256"] = sha.lower()
            for field in ("gt_initial_frames", "gt_additional_frames", "accepted_frames", "proposed_frames", "rejected_frames", "automatic_masked_joints", "consensus_joints", "visible_pseudo_joints", "gt_query_calls", "human_minutes", "training_seconds"):
                if field in audit: item[field] = audit[field]
            rows[k] = item
    return rows, sources, issues


def check_comparability(path):
    required = ("shared_initial_gt_sets", "pretrained_backbone", "training_configuration", "software_environment", "source_split")
    if path is None or not Path(lp(path)).is_file():
        return False, [{"type":"comparability_not_verified", "path":str(path)}]
    record = read_json(path)
    issues = []
    if record.get("status") != "verified":
        issues.append({"type":"comparability_not_verified","status":record.get("status")})
    for name in required:
        r = record.get("checks",{}).get(name,{})
        if r.get("status") != "verified" or not r.get("evidence"):
            issues.append({"type":"comparability_check_incomplete","check":name})
    return not issues, issues


def load_penn(base, protocol):
    groups, sources, issues = {}, [], []
    evaluations = base / "evaluations"
    if not Path(lp(evaluations)).is_dir():
        return groups, sources, [{"type": "missing_input", "path": str(evaluations)}]
    for raw in sorted(Path(lp(evaluations)).glob("*/per_clip_raw.csv")):
        meta_path = raw.parent / "metrics.json"
        if not meta_path.is_file():
            issues.append({"type": "penn_completion_not_verified", "path": str(raw)})
            continue
        meta = read_json(meta_path)
        if meta.get("status") != "complete":
            issues.append({"type": "penn_not_complete", "path": str(meta_path), "status": meta.get("status")})
            continue
        records = read_csv(raw)
        if not records:
            raise ValueError(f"Empty completed external evaluation: {raw}")
        condition = str(records[0].get("condition", records[0].get("method")))
        raw_backbone = str(records[0].get("backbone", meta.get("backbone", ""))).lower()
        canonical = raw_backbone.replace(".pt", "").replace("-pose", "").replace("yolo11", "")
        expected_backbone = str(protocol["backbone"]).lower().replace(".pt", "").replace("-pose", "").replace("yolo11", "")
        if not raw_backbone:
            raise ValueError(f"Missing backbone identity in {raw}")
        if condition not in protocol["methods"] or canonical != expected_backbone:
            issues.append({"type":"outside_planned_external_method_or_backbone", "condition":condition, "backbone":raw_backbone, "path":str(raw)})
            continue
        k = key(records[0])
        if k not in expected_keys(protocol, external=True):
            issues.append({"type": "outside_planned_external_matrix", "key": list(k), "path": str(raw)})
            continue
        if k in groups:
            raise ValueError(f"Duplicate complete external model for {k}")
        clips = {}
        sha_set = set()
        model_ids = set()
        for r in records:
            if key(r) != k or str(r.get("backbone", "")).lower() != raw_backbone:
                raise ValueError(f"Mixed condition/seed/budget/backbone in {raw}")
            clip = str(r["clip_id"])
            if clip in clips:
                raise ValueError(f"Duplicate clip {clip} in {raw}")
            model_ids.add(r["model_id"])
            sha_set.add(r["checkpoint_sha256"].lower())
            if r.get("protocol_sha256") != meta.get("protocol_sha256"):
                raise ValueError(f"Raw/metadata protocol mismatch: {raw}")
            nf = integer(r["n_frames"], "n_frames")
            nd = integer(r["n_frames_with_detection"], "n_frames_with_detection")
            nv = integer(r["n_visible_joints"], "n_visible_joints")
            ne = integer(r["n_evaluable_joints"], "n_evaluable_joints")
            invalid_bbox = integer(r.get("n_invalid_bbox_frames", 0), "n_invalid_bbox_frames")
            if not 0 <= invalid_bbox <= nf:
                raise ValueError(f"Invalid bbox-exclusion count for {clip}")
            c20 = integer(r["correct_020"], "correct_020")
            c10 = integer(r["correct_010"], "correct_010")
            if not (nf > 0 and 0 <= nd <= nf and 0 <= c10 <= c20 <= ne <= nv <= 12 * nf) or c20 > 12 * nd:
                raise ValueError(f"Invalid counts in {raw}, clip {clip}")
            vals = {"pck_020": c20 / ne if ne else None, "pck_010": c10 / ne if ne else None}
            for metric, val in vals.items():
                reported = r.get(metric)
                if val is None:
                    if reported not in (None, "", "nan", "NaN", "null"):
                        raise ValueError("Zero-evaluable clips require missing PCK, not zero")
                elif reported not in (None, "") and not math.isclose(proportion(reported, metric), val, abs_tol=1e-7):
                    raise ValueError(f"Stored {metric} disagrees with counts for {clip}")
            clips[clip] = {"clip_id": clip, "action": r["action"], "n_frames": nf, "n_frames_with_detection": nd, "n_visible_joints": nv, "n_evaluable_joints": ne, "n_invalid_bbox_frames": invalid_bbox, "correct_020": c20, "correct_010": c10, **vals}
        if len(sha_set) != 1 or len(model_ids) != 1:
            raise ValueError(f"Mixed checkpoints/model IDs in {raw}")
        sha = next(iter(sha_set))
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Invalid external checkpoint hash in {raw}")
        if meta.get("checkpoint_sha256") != sha:
            raise ValueError(f"Raw/metadata checkpoint mismatch: {raw}")
        for unit, observed in (("clips", len(clips)), ("frames", sum(r["n_frames"] for r in clips.values()))):
            expected = meta.get(f"official_test_{unit}_expected")
            processed = meta.get(f"official_test_{unit}_processed")
            if expected is None or processed is None or int(expected) != observed or int(processed) != observed or observed != protocol[f"penn_test_{unit}_expected"]:
                raise ValueError(f"{raw}: completed official-test {unit} counts are missing/inconsistent")
        for hash_name in ("protocol_sha256", "prepare_sha256"):
            h = meta.get(hash_name, "")
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h.lower()):
                raise ValueError(f"{raw}: missing/invalid {hash_name}")
        if meta.get("model_id") != next(iter(model_ids)):
            raise ValueError(f"Raw/metadata model ID mismatch: {raw}")
        eligible_clips = [r for r in clips.values() if r["n_evaluable_joints"] > 0]
        if not eligible_clips:
            raise ValueError(f"No eligible clip in completed {raw}")
        recalculated = {
            "clip_macro_pck_020": float(np.mean([r["pck_020"] for r in eligible_clips])),
            "clip_macro_pck_010": float(np.mean([r["pck_010"] for r in eligible_clips])),
            "micro_pck_020": sum(r["correct_020"] for r in eligible_clips) / sum(r["n_evaluable_joints"] for r in eligible_clips),
            "micro_pck_010": sum(r["correct_010"] for r in eligible_clips) / sum(r["n_evaluable_joints"] for r in eligible_clips),
            "detection_rate": sum(r["n_frames_with_detection"] for r in clips.values()) / sum(r["n_frames"] for r in clips.values()),
            "evaluable_joints": sum(r["n_evaluable_joints"] for r in clips.values()),
            "zero_evaluable_clips": len(clips) - len(eligible_clips),
        }
        for metric, value in recalculated.items():
            reported = meta.get("metrics", {}).get(metric)
            if reported is None or not math.isclose(number(reported, metric), value, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(f"Raw/metadata {metric} mismatch: {raw}")
        groups[k] = {"clips": clips, "checkpoint_sha256": sha, "model_id": next(iter(model_ids)), "metadata": meta}
        sources.extend((raw, meta_path))
    if groups:
        first_group = next(iter(groups.values()))
        first = first_group["clips"]
        for k, g in groups.items():
            for h in ("protocol_sha256", "prepare_sha256"):
                if g["metadata"][h] != first_group["metadata"][h]:
                    raise ValueError(f"Different evaluation protocol or prepared test data across models: {k}, {h}")
            if set(g["clips"]) != set(first):
                raise ValueError(f"Different test clips across models: {k}")
            for clip, reference in first.items():
                for field in ("action", "n_frames", "n_visible_joints", "n_evaluable_joints", "n_invalid_bbox_frames"):
                    if g["clips"][clip][field] != reference[field]:
                        raise ValueError(f"Model-dependent GT denominator/metadata: {k}, {clip}, {field}")
    return groups, sources, issues


def summarize_training(rows, p):
    curve, per_auc, auc_summary, differences = [], [], [], []
    for method in p["methods"]:
        for budget in p["budgets"]:
            observed = [r for k, r in rows.items() if k[0] == method and k[2] == budget]
            if not observed:
                continue
            v = np.array([r["map50_95"] for r in observed])
            curve.append({"condition": method, "budget": budget, "n_training_seeds": len(v), "map50_95_mean": float(v.mean()), "training_seed_sd_ddof0": float(v.std(ddof=0)) if len(v) > 1 else None, "minimum": float(v.min()), "maximum": float(v.max())})
        for seed in p["seeds"]:
            observed = [rows[(method, seed, b)] for b in p["budgets"] if (method, seed, b) in rows]
            if len(observed) == len(p["budgets"]):
                auc = normalized_budget_auc([r["budget"] for r in observed], [r["map50_95"] for r in observed], p["budgets"])
                per_auc.append({"condition": method, "seed": seed, "budget_auc_map50_95": auc})
        values = [r["budget_auc_map50_95"] for r in per_auc if r["condition"] == method]
        if values:
            auc_summary.append({"condition": method, "n_training_seeds": len(values), "budget_auc_mean": float(np.mean(values)), "training_seed_sd_ddof0": float(np.std(values)) if len(values) > 1 else None})
    indexed = {(r["condition"], r["seed"]): r["budget_auc_map50_95"] for r in per_auc}
    for a, b in [p["primary_contrast"], *p["other_contrasts"]]:
        for seed in p["seeds"]:
            if (a, seed) in indexed and (b, seed) in indexed:
                differences.append({"condition_a": a, "condition_b": b, "seed": seed, "paired_budget_auc_difference_a_minus_b": indexed[(a, seed)] - indexed[(b, seed)]})
    return curve, per_auc, auc_summary, differences


def summarize_penn(groups, p):
    per_seed, aggregate, comparisons, per_action = [], [], [], []
    vectors = {}
    resamples = p["bootstrap_resamples"]
    bootseed = p["bootstrap_seed"]
    for k, g in sorted(groups.items()):
        eligible = [g["clips"][clip] for clip in sorted(g["clips"]) if g["clips"][clip]["n_evaluable_joints"] > 0]
        if not eligible:
            raise ValueError(f"No evaluable external clips for {k}")
        v20 = np.array([r["pck_020"] for r in eligible])
        v10 = np.array([r["pck_010"] for r in eligible])
        vectors[k] = v20
        ci = paired_clip_bootstrap(v20, n_boot=resamples, seed=bootseed)
        allclips = list(g["clips"].values())
        per_seed.append({"condition": k[0], "seed": k[1], "budget": k[2], "model_id": g["model_id"], "checkpoint_sha256": g["checkpoint_sha256"], "n_test_clips": len(allclips), "n_evaluable_clips": len(eligible), "n_zero_evaluable_clips": len(allclips)-len(eligible), "n_test_frames": sum(r["n_frames"] for r in allclips), "n_frames_with_detection": sum(r["n_frames_with_detection"] for r in allclips), "n_visible_joints": sum(r["n_visible_joints"] for r in allclips), "n_evaluable_joints": sum(r["n_evaluable_joints"] for r in allclips), "n_invalid_bbox_frames": sum(r["n_invalid_bbox_frames"] for r in allclips), "n_excluded_visible_joints": sum(r["n_visible_joints"]-r["n_evaluable_joints"] for r in allclips), "macro_pck_020": ci["estimate"], "conditional_clip_ci_low": ci["ci_low"], "conditional_clip_ci_high": ci["ci_high"], "micro_pck_020": sum(r["correct_020"] for r in eligible)/sum(r["n_evaluable_joints"] for r in eligible), "macro_pck_010": float(v10.mean()), "uncertainty_scope": "clip sampling conditional on one trained checkpoint"})
        for action in sorted({r["action"] for r in eligible}):
            action_values = [r["pck_020"] for r in eligible if r["action"] == action]
            per_action.append({"condition": k[0], "seed": k[1], "budget": k[2], "action": action, "n_evaluable_clips": len(action_values), "macro_pck_020": float(np.mean(action_values))})
    for budget in p["penn_budgets"]:
        for method in p["methods"]:
            seeds = [s for s in p["seeds"] if (method, s, budget) in vectors]
            if not seeds:
                continue
            seed_scores = [float(vectors[(method, s, budget)].mean()) for s in seeds]
            averaged_clips = np.mean([vectors[(method, s, budget)] for s in seeds], axis=0)
            ci = paired_clip_bootstrap(averaged_clips, n_boot=resamples, seed=bootseed)
            aggregate.append({"condition": method, "budget": budget, "n_training_seeds": len(seeds), "training_seeds": ",".join(map(str, seeds)), "macro_pck_020_mean": ci["estimate"], "training_seed_sd_ddof0": float(np.std(seed_scores)) if len(seeds) > 1 else None, "conditional_clip_ci_low": ci["ci_low"], "conditional_clip_ci_high": ci["ci_high"], "n_evaluable_clips": ci["n_clips"], "uncertainty_scope": "clip sampling conditional on the listed trained checkpoints; seed SD separate"})
        for a, b in [p["primary_contrast"], *p["other_contrasts"]]:
            paired_seeds = [s for s in p["seeds"] if (a, s, budget) in vectors and (b, s, budget) in vectors]
            for s in paired_seeds:
                ci = paired_clip_bootstrap(vectors[(a, s, budget)], vectors[(b, s, budget)], n_boot=resamples, seed=bootseed)
                comparisons.append({"condition_a": a, "condition_b": b, "budget": budget, "seed": s, "n_training_seeds": 1, "training_seeds": str(s), **ci, "uncertainty_scope": "paired clips conditional on the two checkpoints"})
            if len(paired_seeds) > 1:
                av = np.mean([vectors[(a,s,budget)] for s in paired_seeds], axis=0)
                bv = np.mean([vectors[(b,s,budget)] for s in paired_seeds], axis=0)
                ci = paired_clip_bootstrap(av, bv, n_boot=resamples, seed=bootseed)
                comparisons.append({"condition_a": a, "condition_b": b, "budget": budget, "seed": "mean_of_paired_seeds", "n_training_seeds": len(paired_seeds), "training_seeds": ",".join(map(str, paired_seeds)), **ci, "uncertainty_scope": "paired clips conditional on these fitted seeds; no training-seed resampling"})
    return per_seed, aggregate, comparisons, per_action


def table_text(title, rows, fields):
    if not rows:
        return title + "\nNo completed observations.\n"
    lines = [title, "\t".join(fields)]
    for r in rows:
        lines.append("\t".join("NA" if r.get(k) is None else (f"{r[k]:.6f}" if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in fields))
    return "\n".join(lines) + "\n"


def plots(output, curve, per_auc, external, comparisons, p, partial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = dict(zip(p["methods"], ["#557A95", "#B44C37", "#3D8662", "#775DA6", "#D19A2B"]))
    suffix = " (partial descriptive results)" if partial else ""
    def emit(fig, name):
        fig.savefig(lp(output / (name + ".png")), dpi=300, bbox_inches="tight")
        fig.savefig(lp(output / (name + ".pdf")), bbox_inches="tight")
        plt.close(fig)
    if curve:
        fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
        for m in p["methods"]:
            vals = sorted([r for r in curve if r["condition"] == m], key=lambda x: x["budget"])
            if vals:
                x = [r["budget"]*100 for r in vals]
                y = [r["map50_95_mean"] for r in vals]
                ax.plot(x, y, marker="o", color=colors[m], label=LABELS[m])
                for a, b, r in zip(x, y, vals):
                    if r["training_seed_sd_ddof0"] is not None:
                        ax.errorbar([a], [b], yerr=[r["training_seed_sd_ddof0"]], color=colors[m], capsize=3)
        ax.set(xlabel="Accepted training frames (% of source pool)", ylabel="Source validation pose mAP@50–95", title="YOLO11n matched follow-up suite"+suffix, ylim=(0,1))
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc="best")
        emit(fig, "source_budget_curves")
    if per_auc:
        fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
        for i, m in enumerate(p["methods"]):
            vals = sorted([r for r in per_auc if r["condition"] == m], key=lambda r:r["seed"])
            for j, r in enumerate(vals):
                ax.scatter(i+(j-(len(vals)-1)/2)*.11, r["budget_auc_map50_95"], color=colors[m], s=44)
            if vals:
                ax.plot([i-.22,i+.22],[np.mean([r["budget_auc_map50_95"] for r in vals])]*2, color="black", linewidth=1.5)
        ax.set_xticks(range(len(p["methods"])), ["A0","Boracle","ref","Joint mask","Flip consensus"])
        ax.set(ylabel="Normalized budget AUC (mAP@50–95)", title="Each point is one complete training-seed curve"+suffix, ylim=(0,1))
        ax.grid(axis="y", alpha=.18)
        emit(fig, "source_budget_auc_per_seed")
    for budget in p["penn_budgets"]:
        vals = [r for r in external if r["budget"] == budget]
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
        for i, m in enumerate(p["methods"]):
            rr = [r for r in vals if r["condition"] == m]
            if not rr: continue
            r = rr[0]
            ax.scatter(r["macro_pck_020_mean"]*100, i, color=colors[m], s=45)
            if r["conditional_clip_ci_low"] is not None:
                ax.hlines(i, r["conditional_clip_ci_low"]*100, r["conditional_clip_ci_high"]*100, color=colors[m], linewidth=2)
        ax.set_yticks(range(len(p["methods"])), [LABELS[m] for m in p["methods"]])
        ax.set(xlabel="Penn Action clip-macro PCK@0.20 (%)", title=f"Source accepted-frame budget {budget:.0%}"+suffix, xlim=(0,100))
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.18)
        fig.text(.5,-.03,"Intervals resample paired test clips; training-seed variability is reported separately.",ha="center",fontsize=8)
        emit(fig, f"penn_action_macro_pck_budget_{budget:.2f}")
        diff = [r for r in comparisons if r["budget"] == budget and r["seed"] == "mean_of_paired_seeds"]
        if diff:
            fig, ax = plt.subplots(figsize=(7.6, 3.5), layout="constrained")
            for i, r in enumerate(diff):
                ax.scatter(r["estimate"]*100, i, color="#37474F")
                if r["ci_low"] is not None: ax.hlines(i,r["ci_low"]*100,r["ci_high"]*100,color="#37474F",linewidth=2)
            ax.set_yticks(range(len(diff)), [f"Boracle minus {LABELS[r['condition_b']]}" for r in diff])
            ax.axvline(0,color="gray",linestyle="--",linewidth=1)
            ax.set(xlabel="Paired difference in clip-macro PCK@0.20 (percentage points)",title="Conditional paired-clip bootstrap contrasts"+suffix)
            ax.invert_yaxis(); ax.grid(axis="x",alpha=.18)
            emit(fig, f"penn_action_paired_differences_budget_{budget:.2f}")


def manuscript_insert(p, curve, aucs, ext, contrasts):
    lines = [
        "ADDITIONAL EXPLORATORY EXPERIMENT — DRAFT INSERT FOR AUTHOR REVIEW",
        "Generated from completed, validated observations. This does not modify the manuscript.",
        "",
        "Methods.",
        "We extended the YOLO11n-pose comparison across six training seeds (0–5), 30 epochs per budget round, a batch size of 16, and 416-pixel training inputs. The original A0/Boracle/ref checkpoints and source-validation measurements were reused, while joint-confidence masking and flip-consensus were newly trained as follow-up arms. Initial GT sets and labels, source split, recorded training settings, retained pretrained backbone, and recorded critical software versions were checked before combining the results. The historical starting-weight digest and full transitive dependency lock were unavailable; historical byte identity and bitwise equivalence are not established. The source split retained REHAB24-6 subjects 1–8 for training and subjects 9–10 for checkpoint selection. The two new arms extend the nano-backbone comparison and are not a fresh rerun of the original three-backbone study. The source validation set and all ten source participants had prior development exposure; the new run does not convert those data into a previously untouched cohort.",
        "The methods were the original self-training procedure after shared GT initialization (A0), a ground-truth-derived correction-plus-rejection oracle (Boracle), a full-GT comparator (ref), joint-confidence masking, and flip-consensus pseudo-labeling. The original confidence-to-visibility conversion in A0 was retained. The source study already included a GT-free frame-level mean-confidence gate (A_conf); the two additional arms extend confidence filtering to joint-level relative-quantile visibility and horizontal-flip agreement. They are custom policy baselines motivated by confidence and consistency principles, rather than reproductions of PoseCons or another published method. After shared initialization, the new arms used no further GT coordinate labels for acquisition or training; source-validation GT remained available for checkpoint selection. All methods used matched accepted-frame budget targets (5%, 10%, 20%, and 40%). Accepted-frame counts do not equate human annotation costs. Boracle-versus-A0 differences represent total adaptive-policy contrasts, including correction, rejection, and changes in subsequent selection paths.",
        "For each new-arm acquisition round, the joint-confidence threshold was the 25th percentile of finite positive keypoint confidences in the remaining unlabeled pool and was held fixed within that round. Retained joints were encoded as visible. This relative threshold had no absolute 0.5 floor and could therefore retain joints that A0 treated as invisible; the policy is not uniformly stricter than A0. Flip-consensus additionally required the original and horizontally flipped predictions to agree within 5% of the image diagonal after inverse flipping and left–right joint remapping, and used their mean coordinates. Both policies rejected frames with fewer than six retained joints or a degenerate retained-joint extent, then proposed additional frames to meet the accepted-frame target. Masking also changed the keypoint-derived training boxes. Their differences from A0 therefore estimate whole pseudo-label-policy contrasts, including visibility, box extent, frame rejection and selection-path changes; the flip arm additionally changes coordinates. The flip arm is two-view agreement postprocessing, not an online consistency-loss or mean-teacher implementation. Round-specific thresholds and accepted-joint counts were recorded for interpretation.",
        "We evaluated the frozen source-trained checkpoints on the official Penn Action test videos without external fine-tuning. The primary external measure was clip-macro PCK@0.20 on 12 common limb joints, using the maximum side of the annotated person box as the normalizing length. Inference used 640-pixel inputs, a detection threshold of 0.25, and the highest-confidence person prediction. Missed detections counted as incorrect for every eligible visible reference joint. Invalid reference exclusions were model-independent and counted explicitly. Penn Action provides cross-dataset action-video evaluation, not rehabilitation-patient validation or a verified subject-independent clinical test.",
        "We report every training seed and its mean and population standard deviation. External 95% percentile intervals used 10,000 paired resamples of test videos, preserving all frames and joints within each video. Seed-averaged intervals first averaged the fitted seeds within each video and then resampled videos. These intervals are conditional on the fitted checkpoints and do not capture training-seed or patient-population uncertainty. Comparisons are exploratory; no multiplicity-adjusted significance or equivalence claim is made. Source label-budget AUC integrates validation mAP over accepted-frame budgets and is distinct from external PCK or a PCK-threshold AUC.",
        "", "Observed results.",
    ]
    for r in aucs:
        lines.append(f"{LABELS[r['condition']]}: normalized source budget AUC {r['budget_auc_mean']:.4f} (population SD {r['training_seed_sd_ddof0']:.4f}; {r['n_training_seeds']} seeds).")
    for r in ext:
        ci = "not estimable with fewer than two eligible videos" if r["conditional_clip_ci_low"] is None else f"{r['conditional_clip_ci_low']*100:.2f}–{r['conditional_clip_ci_high']*100:.2f}%"
        lines.append(f"At source budget {r['budget']:.0%}, {LABELS[r['condition']]} achieved clip-macro PCK@0.20 {r['macro_pck_020_mean']*100:.2f}% (conditional clip-bootstrap 95% CI {ci}; training-seed population SD {r['training_seed_sd_ddof0']*100:.2f} percentage points; {r['n_evaluable_clips']} evaluable videos).")
    for r in contrasts:
        if r["seed"] == "mean_of_paired_seeds":
            ci = "not estimable with fewer than two eligible videos" if r["ci_low"] is None else f"{r['ci_low']*100:+.2f} to {r['ci_high']*100:+.2f} percentage points"
            lines.append(f"At budget {r['budget']:.0%}, {LABELS[r['condition_a']]} minus {LABELS[r['condition_b']]} was {r['estimate']*100:+.2f} percentage points (conditional paired-video 95% CI {ci}).")
    lines += ["", "These observations do not establish real human-correction efficiency, clinical benefit, or patient generalization. The oracle remains dependent on reference labels. A negative or null contrast must be retained as observed; it must not be replaced by a predicted improvement."]
    return ("\n".join(lines) + "\n").replace("10,000", format(p["bootstrap_resamples"], ","))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lock-protocol", type=Path)
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--training-dir", type=Path)
    ap.add_argument("--training-csv", action="append", type=Path, default=[])
    ap.add_argument("--training-audit-jsonl", action="append", type=Path, default=[])
    ap.add_argument("--comparability-receipt", type=Path)
    ap.add_argument("--penn-dir", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--penn-budgets", nargs="+", type=float, default=[.40])
    ap.add_argument("--pool-size", type=int, default=8367)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_tests()
    if args.lock_protocol:
        p = protocol_template(args)
        check_protocol(p)
        Path(lp(args.lock_protocol.parent)).mkdir(parents=True, exist_ok=True)
        save_json(args.lock_protocol, p)
        text_path = args.lock_protocol.with_suffix(".txt")
        save_text(text_path, "분석 규칙 사전 고정(실험 결과 아님)\n" + json.dumps(p, ensure_ascii=False, indent=2) + "\n")
        print(f"분석 규칙 저장: {args.lock_protocol}. 새 결과는 생성하지 않았습니다.")
        return 0
    if not all((args.protocol, args.penn_dir, args.output)) or not (args.training_dir or args.training_csv):
        ap.error("Summarization needs --protocol, --penn-dir, --output, and training input(s)")
    p = read_json(args.protocol)
    check_protocol(p)
    training, train_sources, train_issues = load_training(args.training_dir, p, args.training_csv, args.training_audit_jsonl)
    external, ext_sources, ext_issues = load_penn(args.penn_dir, p)
    if not training and not external:
        print("완료된 실제 실험 결과가 없습니다. 표·그림·영문 삽입문을 생성하지 않았습니다.")
        return 2
    missing_train = sorted(expected_keys(p) - set(training))
    missing_ext = sorted(expected_keys(p, external=True) - set(external))
    comparable, comparison_issues = check_comparability(args.comparability_receipt)
    partial = bool(missing_train or missing_ext or not comparable)
    if partial and not args.allow_partial:
        print(json.dumps({"status":"incomplete", "missing_training":missing_train, "missing_external":missing_ext, "issues":train_issues+ext_issues+comparison_issues},ensure_ascii=False,indent=2))
        print("계획된 비교행렬이 미완료입니다. --allow-partial에서만 부분 기술표를 생성할 수 있습니다.")
        return 2
    for k in set(training) & set(external):
        if training[k]["checkpoint_sha256"] != external[k]["checkpoint_sha256"]:
            raise ValueError(f"Training/external checkpoint hash mismatch for {k}")
    out = Path(lp(args.output))
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is nonempty; choose a new path to preserve previous results")
    out.mkdir(parents=True, exist_ok=True)
    curve, per_auc, auc_summary, auc_diff = summarize_training(training, p)
    ext_per, ext_summary, contrasts, by_action = summarize_penn(external, p)
    if not comparable:
        auc_diff, contrasts = [], []
    for table in (curve, per_auc, auc_summary, ext_per, ext_summary, by_action):
        for row in table:
            row["provenance"] = p["condition_provenance"][row["condition"]]
    tables = {
        "training_completed_raw": list(training.values()), "source_budget_curve_summary": curve,
        "source_budget_auc_per_seed": per_auc, "source_budget_auc_summary": auc_summary,
        "source_budget_auc_paired_seed_differences": auc_diff,
        "penn_action_per_seed": ext_per, "penn_action_summary": ext_summary,
        "penn_action_paired_clip_differences": contrasts, "penn_action_by_action_per_seed": by_action,
    }
    for name, rows in tables.items(): save_csv(out/(name+".csv"), rows)
    status = {"schema_version":SCHEMA_VERSION,"generated_at_utc":now(),"status":"partial_descriptive" if partial else "complete_planned_suite","training_completed":len(training),"external_completed":len(external),"missing_training":[list(k) for k in missing_train],"missing_external":[list(k) for k in missing_ext],"issues":train_issues+ext_issues+comparison_issues,"manuscript_insert_generated":not partial,"comparability_verified":comparable,"analysis_protocol_sha256":digest(args.protocol),"source_files":[{"path":str(q),"sha256":digest(q)} for q in [args.protocol,*train_sources,*ext_sources,*([args.comparability_receipt] if args.comparability_receipt and Path(lp(args.comparability_receipt)).is_file() else [])]],"uncertainty_notes":p["interpretation"],"single_seed_limit":"A single training seed has no estimated training-seed SD; clip CI is conditional on that checkpoint."}
    overview = "추가 실험 분석 결과\n상태: " + status["status"] + "\n\n"
    overview += "원본 A/B/ref를 재사용하고 두 자동 비교군을 새로 학습한 YOLO11n·6시드·batch16 후속 실험입니다. 원본/신규 출처를 각 행에 표시합니다. 단위는 비율(0–1)이며 그림의 PCK만 %입니다.\n"
    overview += "예산 AUC는 source mAP 대 채택 프레임 예산입니다. 외부 PCK와 같은 지표가 아닙니다.\n"
    overview += "PCK 구간은 동일 영상의 paired bootstrap이며 체크포인트 조건부입니다. 학습시드 SD와 모집단 불확실성을 구별합니다.\n\n"
    overview += table_text("Source budget AUC", auc_summary,["condition","n_training_seeds","budget_auc_mean","training_seed_sd_ddof0"])+"\n"
    overview += table_text("Penn Action",ext_summary,["condition","budget","n_training_seeds","macro_pck_020_mean","training_seed_sd_ddof0","conditional_clip_ci_low","conditional_clip_ci_high"])
    if partial: overview += "\n미완료 비교행렬이므로 부분 기술 결과입니다. 영문 원고 삽입문은 생성하지 않았습니다. 누락 목록은 analysis_status.json에 있습니다.\n"
    save_text(out/"결과_읽는법.txt",overview)
    plots(out,curve,per_auc,ext_summary,contrasts,p,partial)
    if not partial: save_text(out/"manuscript_insert_observed_results.txt",manuscript_insert(p,curve,auc_summary,ext_summary,contrasts))
    save_json(out/"analysis_status.json",status)
    print(f"실제 완료 결과 분석: training {len(training)}건, external {len(external)}건. 상태={status['status']}. 저장={args.output}")
    return 0


def run_self_tests():
    class AnalysisTests(unittest.TestCase):
        def test_auc_trapezoid_normalization(self):
            self.assertAlmostEqual(normalized_budget_auc([.05,.1,.2,.4],[.2,.4,.6,.8],[.05,.1,.2,.4]),.205/.35)
        def test_auc_rejects_missing(self):
            with self.assertRaises(ValueError): normalized_budget_auc([.05,.1,.4],[.2,.4,.8],[.05,.1,.2,.4])
        def test_auc_rejects_duplicate(self):
            with self.assertRaises(ValueError): normalized_budget_auc([.05,.1,.1,.4],[.2,.4,.6,.8],[.05,.1,.2,.4])
        def test_paired_identical_is_zero(self):
            result=paired_clip_bootstrap([0,.2,1],[0,.2,1],n_boot=1000)
            self.assertEqual(result["estimate"],0)
            self.assertEqual(result["ci_low"],0)
            self.assertEqual(result["ci_high"],0)
        def test_paired_constant_difference(self):
            result=paired_clip_bootstrap([.3,.5,.9],[.1,.3,.7],n_boot=1000)
            self.assertAlmostEqual(result["estimate"],.2)
            self.assertAlmostEqual(result["ci_low"],.2)
            self.assertAlmostEqual(result["ci_high"],.2)
        def test_one_clip_has_no_interval(self):
            self.assertIsNone(paired_clip_bootstrap([.5])["ci_low"])
        def test_nan_not_silently_dropped(self):
            with self.assertRaises(ValueError): paired_clip_bootstrap([.5,float("nan")])
        def test_pair_length_mismatch(self):
            with self.assertRaises(ValueError): paired_clip_bootstrap([.1,.2],[.1])
        def test_pck_macro_is_not_micro(self):
            self.assertAlmostEqual(np.mean([1/1,0/9]),.5)
            self.assertAlmostEqual((1+0)/(1+9),.1)
        def test_invalid_numeric_value(self):
            with self.assertRaises(ValueError): proportion(90,"PCK")
        def test_no_seed_sd_with_one_seed(self):
            p={"methods":["A"],"seeds":[0],"budgets":[.05,.1,.2,.4],"primary_contrast":["B","A"],"other_contrasts":[]}
            rows={("A",0,b):{"condition":"A","seed":0,"budget":b,"map50_95":.4} for b in p["budgets"]}
            curve,per,summary,diff=summarize_training(rows,p)
            self.assertIsNone(summary[0]["training_seed_sd_ddof0"])
            self.assertEqual(len(per),1)
        def test_incomplete_seed_not_interpolated(self):
            p={"methods":["A"],"seeds":[0],"budgets":[.05,.1,.2,.4],"primary_contrast":["B","A"],"other_contrasts":[]}
            rows={("A",0,.05):{"condition":"A","seed":0,"budget":.05,"map50_95":.4}}
            self.assertEqual(summarize_training(rows,p)[1],[])
        def test_unestimable_interval_is_not_fabricated(self):
            p={"bootstrap_resamples":1000}
            row={"condition":"A","budget":.4,"macro_pck_020_mean":.5,"conditional_clip_ci_low":None,"conditional_clip_ci_high":None,"training_seed_sd_ddof0":.1,"n_evaluable_clips":1}
            txt=manuscript_insert(p,[],[],[row],[])
            self.assertIn("not estimable",txt)
        def test_file_pipeline_synthetic_only(self):
            import tempfile
            from types import SimpleNamespace
            with tempfile.TemporaryDirectory(prefix="rehabal_SYNTHETIC_TEST_") as td:
                root = Path(td)
                p = protocol_template(SimpleNamespace(penn_budgets=[.4],pool_size=8367,bootstrap=1000))
                p["penn_test_clips_expected"] = 2
                p["penn_test_frames_expected"] = 3
                save_json(root/"protocol.json",p)
                train = root/"training"
                train.mkdir()
                rows, audits = [], []
                for method, seed, budget in sorted(expected_keys(p)):
                    h = hashlib.sha256(f"SYNTHETIC {method} {seed} {budget}".encode()).hexdigest()
                    rows.append({"condition":method,"seed":seed,"budget":budget,"sigma":0 if method=="B" else "","n_labeled":round(8367*budget),"map50_95":.3+budget,"map50":.5+budget})
                    audits.append({"condition":method,"seed":seed,"budget":budget,"status":"complete","checkpoint_sha256":h,"provenance":p["condition_provenance"][method]})
                save_csv(train/"clinician_curve.csv",rows)
                save_text(train/"audit_rounds.jsonl","\n".join(json.dumps(r) for r in audits)+"\n")
                hashes={(r["condition"],r["seed"],r["budget"]):r["checkpoint_sha256"] for r in audits}
                for method, seed, budget in sorted(expected_keys(p,external=True)):
                    d=root/"penn"/"evaluations"/f"SYNTHETIC_{method}_{seed}"
                    d.mkdir(parents=True)
                    model=d.name
                    sha=hashes[(method,seed,budget)]
                    common={"model_id":model,"method":method,"seed":seed,"budget":budget,"backbone":"yolo11n-pose","checkpoint_sha256":sha,"protocol_sha256":"a"*64,"action":"SYNTHETIC"}
                    cr=[{**common,"clip_id":"0001","n_frames":1,"n_frames_with_detection":1,"n_visible_joints":12,"n_evaluable_joints":12,"correct_020":6,"correct_010":3,"pck_020":.5,"pck_010":.25}, {**common,"clip_id":"0002","n_frames":2,"n_frames_with_detection":1,"n_visible_joints":24,"n_evaluable_joints":24,"correct_020":6,"correct_010":3,"pck_020":.25,"pck_010":.125}]
                    save_csv(d/"per_clip_raw.csv",cr)
                    meta={"status":"complete","model_id":model,"checkpoint_sha256":sha,"protocol_sha256":"a"*64,"prepare_sha256":"b"*64,"official_test_clips_expected":2,"official_test_clips_processed":2,"official_test_frames_expected":3,"official_test_frames_processed":3,"metrics":{"clip_macro_pck_020":.375,"clip_macro_pck_010":.1875,"micro_pck_020":12/36,"micro_pck_010":6/36,"detection_rate":2/3,"evaluable_joints":36,"zero_evaluable_clips":0}}
                    save_json(d/"metrics.json",meta)
                receipt={"status":"verified","checks":{name:{"status":"verified","evidence":["SYNTHETIC FIXTURE ONLY"]} for name in ("shared_initial_gt_sets","pretrained_backbone","training_configuration","software_environment","source_split")}}
                save_json(root/"receipt.json",receipt)
                loaded,_,issues=load_training(train,p)
                self.assertEqual(len(loaded),120)
                self.assertEqual(issues,[])
                penn,_,issues=load_penn(root/"penn",p)
                self.assertEqual(len(penn),30)
                self.assertEqual(issues,[])
                per,summary,diff,actions=summarize_penn(penn,p)
                self.assertAlmostEqual(summary[0]["macro_pck_020_mean"],.375)
                self.assertEqual(summary[0]["n_training_seeds"],6)
                self.assertTrue(all(r["estimate"]==0 for r in diff))
                source_model=root/"penn"/"evaluations"/next(iter(penn.values()))["model_id"]
                extra_model=root/"penn"/"evaluations"/"SYNTHETIC_secondary_s"
                extra_model.mkdir()
                extra_rows=read_csv(source_model/"per_clip_raw.csv")
                for r in extra_rows:
                    r["backbone"]="yolo11s"
                    r["model_id"]=extra_model.name
                save_csv(extra_model/"per_clip_raw.csv",extra_rows)
                extra_meta=read_json(source_model/"metrics.json")
                extra_meta.update({"backbone":"yolo11s","model_id":extra_model.name})
                save_json(extra_model/"metrics.json",extra_meta)
                primary,_,excluded=load_penn(root/"penn",p)
                self.assertEqual(len(primary),30)
                self.assertEqual(excluded[0]["type"],"outside_planned_external_method_or_backbone")
                self.assertFalse(check_comparability(root/"missing.json")[0])
                one_key=next(iter(penn))
                first_model=root/"penn"/"evaluations"/penn[one_key]["model_id"]
                original_meta=read_json(first_model/"metrics.json")
                damaged=dict(original_meta)
                damaged["protocol_sha256"]="c"*64
                with open(lp(first_model/"metrics.json"),"w",encoding="utf-8") as f: json.dump(damaged,f)
                with self.assertRaises(ValueError): load_penn(root/"penn",p)
                with open(lp(first_model/"metrics.json"),"w",encoding="utf-8") as f: json.dump(original_meta,f)
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    result=main(["--protocol",str(root/"protocol.json"),"--training-dir",str(train),"--penn-dir",str(root/"penn"),"--comparability-receipt",str(root/"receipt.json"),"--output",str(root/"SYNTHETIC_OUTPUT")])
                self.assertEqual(result,0)
                status=read_json(root/"SYNTHETIC_OUTPUT"/"analysis_status.json")
                self.assertEqual(status["status"],"complete_planned_suite")
                self.assertTrue((root/"SYNTHETIC_OUTPUT"/"manuscript_insert_observed_results.txt").is_file())
                self.assertEqual(len(list((root/"SYNTHETIC_OUTPUT").glob("*.png"))),4)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests))
    print("합성 검증: " + ("통과" if result.wasSuccessful() else "실패") + ". 실제 실험 결과는 생성하지 않았으며 임시 합성 자료는 정리했습니다.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"분석 중단: {exc}",file=sys.stderr)
        sys.exit(1)
