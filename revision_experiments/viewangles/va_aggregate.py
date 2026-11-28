import argparse
import csv
import json
from pathlib import Path

import numpy as np

BINS = ["0", "10-30", "40-60", "70-90"]
KEYS = ["detection_rate", "agree010_all", "agree020_all", "agree010_det", "agree020_det", "px_dist_det", "nerr_det"]
CKEYS = ["detection_rate", "agree020_all", "agree020_det", "px_dist_det"]
STRATA = ["overall"] + [f"bin_{b}" for b in BINS] + [f"exercise_{e}" for e in (1, 6, 7)] + [f"camera_{c}" for c in (1, 2, 3)] + \
         [f"exercise_{e}_bin_{b}" for e in (1, 6, 7) for b in BINS]
BOOT_B, BOOT_SEED = 10000, 20261013


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


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


def sel(stratum):
    if stratum == "overall":
        return lambda r: True
    if stratum.startswith("exercise_") and "_bin_" in stratum:
        e, b = stratum.split("_")[1], stratum.split("_bin_")[1]
        return lambda r: r["exercise_id"] == e and r["angle_bin"] == b
    if stratum.startswith("bin_"):
        b = stratum[4:]
        return lambda r: r["angle_bin"] == b
    if stratum.startswith("exercise_"):
        e = stratum.split("_")[1]
        return lambda r: r["exercise_id"] == e
    if stratum.startswith("camera_"):
        c = stratum.split("_")[1]
        return lambda r: r["camera"] == c
    raise ValueError(stratum)


def person_values(rows, k, f):
    by = {}
    for r in rows:
        if f(r):
            by.setdefault(r["person"], []).append(fnum(r[k]))
    return {p: nanmean(v) for p, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluations", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-all", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = [e for e in json.load(open(a.registry, encoding="utf-8-sig"))["models"] if e["budget"] == 0.4 and e["method"] in ("A", "B", "ref")]
    models, missing = {}, []
    for e in reg:
        folder = Path(a.evaluations) / e["model_id"]
        if not (folder / "metrics.json").exists():
            missing.append(e["model_id"])
            continue
        with open(folder / "per_segment.csv", encoding="utf-8") as f:
            seg = list(csv.DictReader(f))
        models[e["model_id"]] = {"entry": e, "metrics": json.load(open(folder / "metrics.json", encoding="utf-8")), "seg": seg}
    if a.require_all and missing:
        raise SystemExit("incomplete: " + ",".join(missing))
    print(json.dumps({"complete": len(models), "missing": missing}))
    if not models:
        return
    per_model = []
    for mid, d in models.items():
        pm = d["metrics"]["metrics"]["person_macro"]
        row = {"model_id": mid, "method": d["entry"]["method"], "seed": d["entry"]["seed"], "frames": d["metrics"]["frames"]}
        for st in STRATA:
            if st in pm:
                for k in KEYS:
                    row[f"{st}.{k}"] = pm[st][k]
        for t, sv in d["metrics"].get("sensitivity_box_conf", {}).items():
            for k in ("detection_rate", "agree020_all", "px_dist_det"):
                row[f"sens_t{t}.{k}"] = sv["person_macro"]["overall"][k]
        per_model.append(row)
    write_csv(out / "per_model.csv", per_model)
    groups = {}
    for mid, d in models.items():
        groups.setdefault(d["entry"]["method"], []).append(mid)
    per_method = []
    for m in ("A", "B", "ref"):
        ids = groups.get(m, [])
        for st in STRATA:
            row = {"method": m, "stratum": st, "n_seeds": len(ids)}
            for k in KEYS:
                vals = np.array([models[i]["metrics"]["metrics"]["person_macro"].get(st, {}).get(k, np.nan) for i in ids], dtype=np.float64)
                row[k + "_mean"] = nanmean(vals)
                row[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else float("nan")
            per_method.append(row)
    write_csv(out / "per_method.csv", per_method)
    sens = []
    for m in ("A", "B", "ref"):
        ids = groups.get(m, [])
        row = {"method": m, "n_seeds": len(ids)}
        for t in ("0.25", "0.05", "0.01"):
            for k in ("detection_rate", "agree020_all", "px_dist_det"):
                if t == "0.25":
                    v = [models[i]["metrics"]["metrics"]["person_macro"]["overall"][k] for i in ids]
                else:
                    v = [models[i]["metrics"]["sensitivity_box_conf"][t]["person_macro"]["overall"][k] for i in ids]
                row[f"t{t}.{k}_mean"] = nanmean(v)
            for b in BINS:
                if t == "0.25":
                    v = [models[i]["metrics"]["metrics"]["person_macro"][f"bin_{b}"]["detection_rate"] for i in ids]
                else:
                    v = [models[i]["metrics"]["sensitivity_box_conf"][t]["person_macro"][f"bin_{b}"]["detection_rate"] for i in ids]
                row[f"t{t}.bin_{b}.detection_rate_mean"] = nanmean(v)
        sens.append(row)
    write_csv(out / "per_method_sensitivity.csv", sens)
    rng = np.random.default_rng(BOOT_SEED)
    base = {models[i]["entry"]["seed"]: i for i in groups.get("A", [])}
    contrasts = []
    for m in ("B", "ref"):
        ids = {models[i]["entry"]["seed"]: i for i in groups.get(m, [])}
        seeds = sorted(set(ids) & set(base))
        for st in ["overall"] + [f"bin_{b}" for b in BINS]:
            f = sel(st)
            for k in CKEYS:
                per_seed, pd = [], {}
                for s in seeds:
                    x = person_values(models[ids[s]]["seg"], k, f)
                    y = person_values(models[base[s]]["seg"], k, f)
                    d = {p: x[p] - y[p] for p in set(x) & set(y) if np.isfinite(x[p]) and np.isfinite(y[p])}
                    per_seed.append(nanmean(list(d.values())))
                    for p, v in d.items():
                        pd.setdefault(p, []).append(v)
                dv = np.array([nanmean(v) for v in pd.values()], dtype=np.float64)
                dv = dv[np.isfinite(dv)]
                if not len(dv):
                    continue
                means = dv[rng.integers(0, len(dv), size=(BOOT_B, len(dv)))].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                contrasts.append({"contrast": f"{m} - A", "stratum": st, "metric": k, "n_persons": int(len(dv)), "n_seeds": len(seeds),
                                  "mean_diff": float(dv.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
                                  "per_seed_diffs": ";".join(f"{v:+.4f}" for v in per_seed), "seeds_positive": int(sum(v > 0 for v in per_seed)),
                                  "note": "person-level bootstrap with n=5 persons; indicative only"})
    write_csv(out / "contrasts_vs_A.csv", contrasts)
    long_rows = []
    for mid, d in models.items():
        for r in d["seg"]:
            long_rows.append({"model_id": mid, **r})
    write_csv(out / "per_segment_all_models.csv", long_rows)
    json.dump({"complete_models": sorted(models), "missing": missing, "reference_is_ground_truth": False, "license": "CC BY-NC 4.0",
               "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "person (n=5)"}}, open(out / "summary.json", "w", encoding="utf-8"), indent=2)
    for row in per_method:
        if row["stratum"].startswith("bin_") or row["stratum"] == "overall":
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in ("method", "stratum", "detection_rate_mean", "agree020_all_mean", "px_dist_det_mean")}))


if __name__ == "__main__":
    main()
