import sys, glob, os
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
suite_dir, grid_csv, e1_csv, cv_csv = sys.argv[1:5]
tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def auc(g):
    g = g.sort_values("budget"); x = g.budget.values; y = g.map50_95.values
    return tz(y, x) / (x[-1] - x[0])


def boot_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


print("=" * 20, "1. strengthened suite + original grid (c17 val, YOLO11n)")
rows = []
for f in sorted(glob.glob(os.path.join(suite_dir, "*", "experiment_curve.csv"))):
    d = pd.read_csv(f); rows.append(d[["condition", "seed", "budget", "map50_95"]])
suite = pd.concat(rows)
grid = pd.read_csv(grid_csv)
grid = grid[(grid.condition.isin(["A", "ref"])) | ((grid.condition == "B") & (grid.sigma == 0))][["condition", "seed", "budget", "map50_95"]]
allc = pd.concat([grid, suite])
tab = allc.groupby(["condition", "budget"]).map50_95.agg(["mean", "std", "count"]).round(3)
print(tab.unstack("budget").to_string())
aucs = allc.groupby(["condition", "seed"])[["budget", "map50_95"]].apply(auc).rename("auc").reset_index()
piv = aucs.pivot(index="seed", columns="condition", values="auc")
print("\nnormalized AUC mean±std:\n", piv.agg(["mean", "std"]).round(3).T.to_string())
for c in ["A_jointmask", "A_flipcons", "B", "ref"]:
    if c in piv:
        d = piv[c] - piv["A"]; lo, hi = boot_ci(d.values)
        gap = (piv[c] - piv["A"]) / (piv["ref"] - piv["A"])
        print(f"  {c:12s} - A: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   gap closure {gap.mean()*100:5.1f}%")
print("\nrejection counts per round (strengthened arms, mean over seeds):")
rej = []
for f in sorted(glob.glob(os.path.join(suite_dir, "*", "experiment_curve.csv"))):
    d = pd.read_csv(f); rej.append(d[["condition", "seed", "budget", "n_proposed", "n_accepted", "n_rejected"]])
rej = pd.concat(rej); print(rej.groupby(["condition", "budget"])[["n_proposed", "n_accepted", "n_rejected"]].mean().round(0).to_string())

print("\n" + "=" * 20, "2. E1 same-frame label-source factorial (frames fixed to ref's per-round lists)")
e1 = pd.read_csv(e1_csv)
cv = pd.read_csv(cv_csv)
ref_rows = cv[cv.condition == "ref"][["seed", "round", "budget", "cam", "map50_95"]].assign(arm="gt (=ref, same frames)")
a_rows = cv[cv.condition == "A"][["seed", "round", "budget", "cam", "map50_95"]].assign(arm="A adaptive (own frames)")
e1all = pd.concat([e1[["arm", "seed", "round", "budget", "cam", "map50_95"]], ref_rows, a_rows])
e1all = e1all[e1all["round"] >= 1]
for cam in ("c17", "c18"):
    sub = e1all[e1all.cam == cam]
    print(f"\n-- {cam}: mAP50-95 mean ± std over seeds (n)")
    t = sub.groupby(["arm", "budget"]).map50_95.agg(["mean", "std", "count"]).round(3)
    print(t.unstack("budget").to_string())
    p = sub.pivot_table(index=["seed", "budget"], columns="arm", values="map50_95")
    print(f"   paired difference vs gt (same frames), mean [95% seed-bootstrap CI]:")
    for arm in [a for a in p.columns if a != "gt (=ref, same frames)"]:
        d = (p[arm] - p["gt (=ref, same frames)"]).dropna()
        for b in sorted(d.index.get_level_values("budget").unique()):
            v = d.xs(b, level="budget").values
            if len(v) >= 2:
                lo, hi = boot_ci(v); print(f"     {arm:26s} budget {b:.2f}: {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(v)}")
print("\nE1 rows:", len(e1), "| sessions complete:", e1.groupby(['arm','seed','round']).ngroups)
