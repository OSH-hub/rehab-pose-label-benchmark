import sys, numpy as np, pandas as pd
csv = sys.argv[1] if len(sys.argv) > 1 else r"rehabal_work\crossview\crossview_11n.csv"
d = pd.read_csv(csv)
B = [0.05, 0.10, 0.20, 0.40]
def auc(g):
    g = g.sort_values("budget"); y = g.map50_95.values; x = g.budget.values
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x) / (x[-1] - x[0])
pd.set_option("display.width", 200)
print("== rows", len(d), "sessions", d.session.nunique())
tab = d.groupby(["cam", "condition", "budget"]).map50_95.agg(["mean", "std", "count"]).round(3)
print(tab.to_string())
aucs = d.groupby(["cam", "condition", "seed"])[["budget", "map50_95"]].apply(auc).rename("auc").reset_index()
m = aucs.groupby(["cam", "condition"]).auc.agg(["mean", "std"]).round(3); print("\n== normalized AUC\n", m.to_string())
for cam in sorted(d.cam.unique()):
    a = aucs[aucs.cam == cam].pivot(index="seed", columns="condition", values="auc")
    if {"A", "B", "ref"} <= set(a.columns):
        gap = (a.B - a.A) / (a.ref - a.A)
        rng = np.random.default_rng(0); boots = [gap.values[rng.integers(0, len(gap), len(gap))].mean() for _ in range(10000)]
        print(f"\n== {cam}: B gap closure mean={gap.mean():.3f}  95% seed-bootstrap CI=[{np.percentile(boots,2.5):.3f}, {np.percentile(boots,97.5):.3f}]  per-seed={np.round(gap.values,3).tolist()}")
        for c1, c2 in (("B", "A"), ("ref", "A"), ("B", "ref")):
            diff = a[c1] - a[c2]; boots = [diff.values[rng.integers(0, 6, 6)].mean() for _ in range(10000)]
            print(f"   AUC {c1}-{c2}: {diff.mean():+.3f} [{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}]")
if {"c17", "c18"} <= set(d.cam.unique()):
    p = d.pivot_table(index=["condition", "seed", "budget"], columns="cam", values="map50_95").reset_index()
    p["drop"] = p.c17 - p.c18; p["rel"] = p["drop"] / p.c17
    print("\n== paired c17->c18 drop in mAP50-95 (mean over seeds)")
    print(p.groupby(["condition", "budget"])[["c17", "c18", "drop", "rel"]].mean().round(3).to_string())
