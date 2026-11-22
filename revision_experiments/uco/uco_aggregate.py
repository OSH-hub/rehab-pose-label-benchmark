r"""Aggregate UCO evaluations across models: per-model table, per-method mean +/- SD over seeds,
subject-level paired bootstrap contrasts vs A (b040), per-camera / per-body strata, out-of-image
GT with/without, and pre-registered box-confidence sensitivity. Reads only completed metrics.json
and per_clip.csv files; never predicts.

Usage: python uco_aggregate.py --evaluations <root> --registry <json> --out <dir> [--require-all]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

BOOT_B = 10000
BOOT_SEED = 20261013
METRICS = ["detection_rate", "angle_mae_det", "angle_median_det", "angle_acc10_all", "angle_acc15_all", "angle_acc10_det",
           "pck02_all", "pck02_det", "pck02_all_vertex", "px_err_det"]
CONTRAST_METRICS = ["detection_rate", "angle_mae_det", "angle_acc10_all", "angle_acc15_all", "pck02_all", "pck02_det", "px_err_det"]
METHOD_ORDER = [("A", 0.4), ("B", 0.4), ("ref", 0.4), ("A", 0.05), ("A_jointmask", 0.4), ("A_flipcons", 0.4)]
STRATA = ["overall"] + [f"camera_{c}" for c in range(5)] + ["body_lower", "body_upper"] + \
         [f"body_{b}_camera_{c}" for b in ("lower", "upper") for c in range(5)]


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def read_clip_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def subject_values(clip_rows, metric, selector=lambda r: True):
    """subject -> mean over the subject's selected clips (NaN-aware)."""
    by = {}
    for r in clip_rows:
        if selector(r):
            by.setdefault(int(r["subject_id"]), []).append(fnum(r[metric]))
    return {s: nanmean(v) for s, v in by.items()}


def selector_for(stratum):
    if stratum == "overall":
        return lambda r: True
    if stratum.startswith("body_") and "camera" in stratum:
        b, c = stratum.split("_")[1], int(stratum.split("_")[3])
        return lambda r: r["body"] == b and int(r["camera_id"]) == c
    if stratum.startswith("camera_"):
        c = int(stratum.split("_")[1])
        return lambda r: int(r["camera_id"]) == c
    if stratum.startswith("body_"):
        b = stratum.split("_")[1]
        return lambda r: r["body"] == b
    raise ValueError(stratum)


def method_key(e):
    return (e["method"], float(e["budget"]))


def method_label(m, b):
    return f"{m}_b{int(round(b * 100)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluations", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-all", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = json.load(open(a.registry, encoding="utf-8-sig"))["models"]
    models = {}
    missing = []
    for e in reg:
        folder = Path(a.evaluations) / e["model_id"]
        if not (folder / "metrics.json").exists():
            missing.append(e["model_id"])
            continue
        m = json.load(open(folder / "metrics.json", encoding="utf-8"))
        models[e["model_id"]] = {"entry": e, "metrics": m, "clips": read_clip_csv(folder / "per_clip.csv")}
    if a.require_all and missing:
        raise SystemExit("incomplete models: " + ",".join(missing))
    print(json.dumps({"complete_models": len(models), "missing": missing}))
    if not models:
        return
    # ---- per-model table (subject-macro, all strata) ----
    per_model = []
    for mid, d in models.items():
        sm = d["metrics"]["metrics"]["subject_macro"]
        row = {"model_id": mid, "method": d["entry"]["method"], "seed": d["entry"]["seed"], "budget": d["entry"]["budget"],
               "frames": d["metrics"]["frames"], "clips": d["metrics"]["clips"]}
        for st in STRATA:
            for k in METRICS:
                row[f"{st}.{k}"] = sm[st][k]
        for k in ("detection_rate", "angle_mae_det", "angle_acc10_all", "pck02_all"):
            row[f"overall_xoob.{k}"] = d["metrics"]["metrics"]["subject_macro_excluding_oob_frames"]["overall"][k]
        for t, sv in d["metrics"].get("sensitivity_box_conf", {}).items():
            for k in ("detection_rate", "angle_mae_det", "angle_acc10_all", "pck02_all", "px_err_det"):
                row[f"sens_t{t}.{k}"] = sv["subject_macro"]["overall"][k]
        per_model.append(row)
    write_csv(out / "per_model.csv", per_model)
    # ---- per-method mean +/- SD over seeds ----
    groups = {}
    for mid, d in models.items():
        groups.setdefault(method_key(d["entry"]), []).append(mid)
    per_method = []
    for (m, b) in METHOD_ORDER:
        ids = groups.get((m, b), [])
        if not ids:
            continue
        for st in STRATA:
            row = {"method": m, "budget": b, "label": method_label(m, b), "stratum": st, "n_seeds": len(ids),
                   "seeds": ",".join(str(models[i]["entry"]["seed"]) for i in ids)}
            for k in METRICS:
                vals = np.array([models[i]["metrics"]["metrics"]["subject_macro"][st][k] for i in ids], dtype=np.float64)
                row[k + "_mean"] = nanmean(vals)
                row[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")
            per_method.append(row)
    write_csv(out / "per_method.csv", per_method)
    # ---- xoob and sensitivity per method (overall) ----
    extra = []
    for (m, b) in METHOD_ORDER:
        ids = groups.get((m, b), [])
        if not ids:
            continue
        row = {"method": m, "budget": b, "label": method_label(m, b), "n_seeds": len(ids)}
        for k in ("detection_rate", "angle_mae_det", "angle_acc10_all", "pck02_all"):
            v_in = [models[i]["metrics"]["metrics"]["subject_macro"]["overall"][k] for i in ids]
            v_x = [models[i]["metrics"]["metrics"]["subject_macro_excluding_oob_frames"]["overall"][k] for i in ids]
            row[f"{k}_incl_oob_mean"] = nanmean(v_in)
            row[f"{k}_excl_oob_mean"] = nanmean(v_x)
        for t in ("0.25", "0.05", "0.01"):
            for k in ("detection_rate", "angle_mae_det", "angle_acc10_all", "pck02_all", "px_err_det"):
                if t == "0.25":
                    v = [models[i]["metrics"]["metrics"]["subject_macro"]["overall"][k] for i in ids]
                else:
                    v = [models[i]["metrics"]["sensitivity_box_conf"][t]["subject_macro"]["overall"][k] for i in ids]
                row[f"t{t}.{k}_mean"] = nanmean(v)
                row[f"t{t}.{k}_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
        extra.append(row)
    write_csv(out / "per_method_oob_and_sensitivity.csv", extra)
    # ---- subject-level paired bootstrap contrasts vs A b040 ----
    rng = np.random.default_rng(BOOT_SEED)
    base_ids = {models[i]["entry"]["seed"]: i for i in groups.get(("A", 0.4), [])}
    contrasts = []
    for (m, b) in METHOD_ORDER:
        if (m, b) == ("A", 0.4):
            continue
        ids = {models[i]["entry"]["seed"]: i for i in groups.get((m, b), [])}
        seeds = sorted(set(ids) & set(base_ids))
        if not seeds:
            continue
        for st in ["overall", "body_lower", "body_upper"] + [f"camera_{c}" for c in range(5)]:
            sel = selector_for(st)
            for k in CONTRAST_METRICS:
                per_seed = []
                subj_diff = {}
                for s in seeds:
                    x = subject_values(models[ids[s]]["clips"], k, sel)
                    y = subject_values(models[base_ids[s]]["clips"], k, sel)
                    common = sorted(set(x) & set(y))
                    d = {sub: x[sub] - y[sub] for sub in common if np.isfinite(x[sub]) and np.isfinite(y[sub])}
                    per_seed.append(nanmean(list(d.values())))
                    for sub, v in d.items():
                        subj_diff.setdefault(sub, []).append(v)
                subs = sorted(subj_diff)
                dv = np.array([nanmean(subj_diff[s]) for s in subs], dtype=np.float64)
                ok = np.isfinite(dv)
                dv = dv[ok]
                n = len(dv)
                if n == 0:
                    continue
                boot = rng.integers(0, n, size=(BOOT_B, n))
                means = dv[boot].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                contrasts.append({"contrast": f"{method_label(m, b)} - A_b040", "stratum": st, "metric": k, "n_subjects": n,
                                  "n_seeds": len(seeds), "mean_diff": float(dv.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi),
                                  "excludes_zero": bool(lo > 0 or hi < 0), "per_seed_diffs": ";".join(f"{v:+.4f}" for v in per_seed),
                                  "seeds_positive": int(sum(1 for v in per_seed if v > 0)), "bootstrap_B": BOOT_B, "bootstrap_seed": BOOT_SEED})
    if contrasts:
        write_csv(out / "contrasts_vs_A.csv", contrasts)
    # ---- clip-level long table of all models (for figures) ----
    long_rows = []
    for mid, d in models.items():
        for r in d["clips"]:
            long_rows.append({"model_id": mid, "method": d["entry"]["method"], "seed": d["entry"]["seed"], "budget": d["entry"]["budget"],
                              **{k: r[k] for k in ("tag", "subject_id", "exercise_id", "camera_id", "body", "side", "n_frames", "n_detected",
                                                   "detection_rate", "angle_mae_det", "angle_acc10_all", "pck02_all", "pck02_det", "px_err_det", "n_oob_frames")}})
    write_csv(out / "per_clip_all_models.csv", long_rows)
    summary = {"complete_models": sorted(models), "missing_models": missing, "metrics": METRICS, "contrast_metrics": CONTRAST_METRICS,
               "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "subject", "pairing": "seed-paired then averaged over seeds"},
               "protocol_sha256": sorted({d["metrics"]["protocol_sha256"] for d in models.values()}),
               "evaluator_sha256": sorted({d["metrics"]["evaluator_sha256"] for d in models.values()}),
               "manifest_sha256": sorted({d["metrics"]["manifest_sha256"] for d in models.values()}),
               "frames": sorted({d["metrics"]["frames"] for d in models.values()}), "clips": sorted({d["metrics"]["clips"] for d in models.values()})}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for row in per_method:
        if row["stratum"] == "overall":
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in ("label", "n_seeds", "detection_rate_mean", "angle_mae_det_mean", "angle_acc10_all_mean", "pck02_all_mean", "px_err_det_mean")}))


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
