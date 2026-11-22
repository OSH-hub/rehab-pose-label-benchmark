r"""Publication figures + tables for the RehabAL-Pose revision (CPU only, matplotlib/pandas).

Every number is computed from the result CSV/JSON files listed in INPUTS below.
Statistics follow the reference analysis scripts shipped with the results
(analyze_e1_and_suite.py, analyze_crossview.py):
  * per-budget mean +- SD over seeds (SD = sample SD, ddof=1, pandas default);
  * normalized label-efficiency AUC per seed = trapezoid(mAP50-95 vs budget) / (b_max - b_min);
  * 95% CI = percentile seed-bootstrap (10,000 resamples, numpy default_rng(0)) of the
    mean of per-seed values (paired differences are formed per seed before resampling);
  * gap closure = mean over seeds of (AUC_cond - AUC_A) / (AUC_ref - AUC_A).
The script also writes sanity_check.txt comparing its numbers against the two
reference outputs (analysis_output_final.txt, crossview_analysis_output.txt).

Run:  python make_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# INPUTS
# --------------------------------------------------------------------------- #
# Root of the revision-experiment outputs and of the core package outputs.
# Set the two environment variables or edit the defaults before running.
SHARED = Path(os.environ.get("REHABAL_REVISION_ROOT", "revision_outputs"))
CODEPKG = Path(os.environ.get("REHABAL_CODE_ROOT", "."))
INPUTS = {
    "suite_dir": SHARED / "training" / "strengthened_suite",                 # <arm>_s<seed>/experiment_curve.csv
    "grid_csv": CODEPKG / "outputs" / "experiments" / "clinician_i4_uniform30" / "clinician_curve.csv",
    "conf_csv": CODEPKG / "outputs" / "experiments" / "clinician_i3_conf" / "clinician_curve.csv",
    "e1_csv": SHARED / "결과_E1_sameframe" / "sameframe_e1.csv",
    "e1_ref_txt": SHARED / "결과_E1_sameframe" / "analysis_output_final.txt",
    "cv_csv": SHARED / "결과_crossview_c18" / "crossview_11n.csv",
    "cv_ref_txt": SHARED / "결과_crossview_c18" / "crossview_analysis_output.txt",
    "cv640_csv": SHARED / "결과_crossview_c18" / "crossview_11n_imgsz640.csv",
    "diag640_txt": SHARED / "결과_crossview_c18" / "imgsz640_진단_20261013.txt",
    # c17 downstream: authoritative corrected run (2026-08-09, same post-correction evaluation code as the c18 run)
    "down_c17_dir": SHARED / "결과_crossview_c18" / "downstream_11n_u30_c17_corrected_2026-08-09",
    # identical copy of the same corrected run inside the code package (used only if the synced
    # placeholders above are not hydrated; verified cell-by-cell against down_c17_md in sanity_check.txt)
    "down_c17_dir_fallback": CODEPKG / "outputs" / "experiments" / "downstream_11n_u30",
    "down_c17_md": SHARED / "결과_crossview_c18" / "downstream_c17_corrected0809.md",   # reference table of that run
    "down_c18_dir": SHARED / "결과_crossview_c18" / "downstream_11n_u30_c18",            # per-seed JSON (c18)
    "down_c18_md": SHARED / "결과_crossview_c18" / "downstream_c18.md",
    # NOT used: downstream_11n_u30_c17/ + downstream_c17.md = older 2026-07-05 run (provenance only)
}
OUT = SHARED / "결과_figures_20261013"

N_BOOT = 10_000
BUDGETS = [0.05, 0.10, 0.20, 0.40]

# --------------------------------------------------------------------------- #
# style (shared with the manuscript's _figstyle.py: Okabe-Ito, distinct ls/marker)
# --------------------------------------------------------------------------- #
COND_STYLE = {
    "A":           dict(color="#0072B2", linestyle="--", marker="s"),               # blue
    "B":           dict(color="#D55E00", linestyle="-",  marker="o"),               # vermillion
    "ref":         dict(color="#009E73", linestyle=":",  marker="^"),               # bluish green
    "A_jointmask": dict(color="#CC79A7", linestyle="-.", marker="D"),               # reddish purple (new)
    "A_flipcons":  dict(color="#E69F00", linestyle=(0, (3, 1, 1, 1, 1, 1)), marker="v"),  # orange (new)
}
COND_LABEL = {
    "A": "A: pseudo-label (GT-free after 5% init)",
    "A_jointmask": "A_jointmask: + joint-confidence masking",
    "A_flipcons": "A_flipcons: + flip-consistency filter",
    "B": "B: simulated correction + reject",
    "ref": "ref: full GT",
}
# sequential scheme (dark -> light = GT-like -> pseudo) for the E1 label-source arms
E1_ARMS = ["gt", "gt_sigma20", "pseudo_init", "pseudo_A"]
E1_LABEL = {
    "gt": "GT labels (= ref, same frames)",
    "gt_sigma20": "GT + N(0, 20 px) noise",
    "pseudo_init": "pseudo-labels, frozen 5% initial model",
    "pseudo_A": "pseudo-labels, self-updating teacher (A)",
    "A_adaptive": "original A (own adaptively selected frames)",
}
E1_STYLE = {
    "gt":          dict(color="#08306B", linestyle="-",  marker="^"),
    "gt_sigma20":  dict(color="#2171B5", linestyle="-",  marker="o"),
    "pseudo_init": dict(color="#6BAED6", linestyle="-",  marker="D"),
    "pseudo_A":    dict(color="#9ECAE1", linestyle="-",  marker="s"),
    "A_adaptive":  dict(color="#555555", linestyle="--", marker="x"),
}
RC = {
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300,
}


def save_fig(plt, fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name + ".png/.pdf")


# --------------------------------------------------------------------------- #
# statistics (identical to the reference scripts)
# --------------------------------------------------------------------------- #
def auc_norm(g):
    g = g.sort_values("budget")
    x = g.budget.values.astype(float); y = g.map50_95.values.astype(float)
    return np.trapezoid(y, x) / (x[-1] - x[0])


def boot_ci(v, n=N_BOOT, seed=0, rng=None):
    """95% percentile bootstrap CI of the mean of v (fresh default_rng(seed) unless rng given)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    v = np.asarray(v, float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def per_seed_auc(df, keys):
    return (df.groupby(keys + ["seed"])[["budget", "map50_95"]].apply(auc_norm)
            .rename("auc").reset_index())


def ms(df, keys):
    return df.groupby(keys).map50_95.agg(["mean", "std", "count"]).reset_index()


def fmt_pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_suite():
    rows = []
    for f in sorted(glob.glob(str(INPUTS["suite_dir"] / "*" / "experiment_curve.csv"))):
        d = pd.read_csv(f)
        rows.append(d[["condition", "seed", "budget", "map50", "map50_95",
                       "n_proposed", "n_accepted", "n_rejected"]])
    suite = pd.concat(rows, ignore_index=True)
    assert suite.groupby("condition").seed.nunique().eq(6).all(), suite.groupby("condition").seed.nunique()
    return suite


def load_grid():
    g = pd.read_csv(INPUTS["grid_csv"])
    g = g[(g.condition.isin(["A", "ref"])) | ((g.condition == "B") & (g.sigma == 0))]
    return g[["condition", "seed", "budget", "map50", "map50_95", "n_proposed", "n_accepted", "n_rejected"]].copy()


def load_downstream(ddir):
    """rows: cond, seed, budget, joint_angle_mae, rom_error, jerk_error."""
    rows = []
    for f in sorted(Path(ddir).glob("*_downstream.json")):
        cond, _, seed = f.stem.replace("_downstream", "").rpartition("_s")
        d = json.loads(f.read_text(encoding="utf-8"))
        for bv in d["budgets"].values():
            if "error" in bv:
                raise ValueError(f"{f.name}: {bv['error']}")
            agg = bv["aggregate"]
            rows.append(dict(cond=cond, seed=int(seed), budget=float(bv["budget"]),
                             joint_angle_mae=agg["joint_angle_mae_mean"]["mean"],
                             rom_error=agg["rom_error_mean"]["mean"],
                             jerk_error=agg["smoothness_error_mean"]))
    return pd.DataFrame(rows)


def parse_downstream_md(path):
    """downstream_c1?.md -> rows: cond, budget, metric, mean, std (SD as printed = population SD, ddof=0)."""
    metric = None; rows = []
    mmap = {"Joint-angle MAE": "joint_angle_mae", "ROM error": "rom_error", "jerk": "jerk_error"}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            metric = next((v for k, v in mmap.items() if k in line), None)
            continue
        m = re.match(r"^\|\s*([0-9.]+)\s*\|(.+)\|\s*$", line)
        if m and metric:
            cells = [c.strip() for c in m.group(2).split("|")]
            for cond, cell in zip(("A", "B", "ref"), cells):
                mu, sd = re.match(r"([0-9.]+)\s*±\s*([0-9.]+)", cell).groups()
                rows.append(dict(cond=cond, budget=float(m.group(1)), metric=metric, mean=float(mu), std=float(sd)))
    df = pd.DataFrame(rows)
    assert len(df) == 36, (path, len(df))
    return df


def parse_diag640(path):
    """imgsz=416 rect=True : pose mAP50-95 0.3273, box mAP50 0.981 -> rows."""
    rows = []
    pat = re.compile(r"imgsz=(\d+)\s+rect=(True|False)\s*:\s*(?:pose mAP50-95\s*)?([0-9.]+)\s*[,/]\s*(?:box mAP50\s*)?([0-9.]+)")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            rows.append(dict(imgsz=int(m.group(1)), rect=(m.group(2) == "True"),
                             pose_map50_95=float(m.group(3)), box_map50=float(m.group(4))))
    df = pd.DataFrame(rows)
    assert len(df) >= 4, df
    return df


# --------------------------------------------------------------------------- #
# analysis 1: strengthened suite
# --------------------------------------------------------------------------- #
def analysis_suite(suite, grid):
    allc = pd.concat([grid, suite[grid.columns]], ignore_index=True)
    conds = ["A", "A_jointmask", "A_flipcons", "B", "ref"]
    tab = ms(allc, ["condition", "budget"])
    aucs = per_seed_auc(allc, ["condition"])
    piv = aucs.pivot(index="seed", columns="condition", values="auc")
    res = {"tab": tab, "aucs": aucs, "piv": piv, "allc": allc}
    auc_rows = []
    for c in conds:
        v = piv[c].values
        lo, hi = boot_ci(v)
        row = dict(condition=c, auc_mean=v.mean(), auc_sd=v.std(ddof=1), auc_ci_lo=lo, auc_ci_hi=hi)
        if c != "A":
            d = (piv[c] - piv["A"]).values
            dlo, dhi = boot_ci(d)
            gap = ((piv[c] - piv["A"]) / (piv["ref"] - piv["A"])).values
            glo, ghi = boot_ci(gap)
            row.update(dauc_vs_A=d.mean(), dauc_ci_lo=dlo, dauc_ci_hi=dhi,
                       gap_closure=gap.mean(), gap_ci_lo=glo, gap_ci_hi=ghi)
        else:
            row.update(dauc_vs_A=0.0, dauc_ci_lo=np.nan, dauc_ci_hi=np.nan,
                       gap_closure=0.0, gap_ci_lo=np.nan, gap_ci_hi=np.nan)
        auc_rows.append(row)
    res["auc_tab"] = pd.DataFrame(auc_rows)
    return res


def fig_strengthened_suite(plt, res):
    tab, auc_tab = res["tab"], res["auc_tab"]
    conds = ["A", "A_jointmask", "A_flipcons", "B", "ref"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for c in conds:
        t = tab[tab.condition == c].sort_values("budget")
        st = COND_STYLE[c]
        lw = 2.4 if c in ("B", "ref") else 1.9
        ax1.errorbar(t.budget, t["mean"], yerr=t["std"], capsize=3, linewidth=lw,
                     markersize=6, label=COND_LABEL[c], **st)
    ax1.set_xscale("log")
    ax1.set_xticks(BUDGETS); ax1.set_xticklabels([f"{b:.2f}" for b in BUDGETS])
    ax1.minorticks_off()
    ax1.set_xlabel("Label budget (fraction of training pool)")
    ax1.set_ylabel("mAP@[.50:.95] (Camera17 validation)")
    ax1.set_title("Label-efficiency curves (YOLO11n, 6 seeds)")
    ax1.legend(loc="upper left", framealpha=1.0)
    ax1.text(0.02, 0.04, "(a)", transform=ax1.transAxes, fontweight="bold")

    x = np.arange(len(conds))
    vals = [float(auc_tab.loc[auc_tab.condition == c, "auc_mean"].iloc[0]) for c in conds]
    lo = [float(auc_tab.loc[auc_tab.condition == c, "auc_ci_lo"].iloc[0]) for c in conds]
    hi = [float(auc_tab.loc[auc_tab.condition == c, "auc_ci_hi"].iloc[0]) for c in conds]
    colors = [COND_STYLE[c]["color"] for c in conds]
    ax2.bar(x, vals, color=colors, width=0.62, edgecolor="black", linewidth=0.6, zorder=3)
    ax2.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                 fmt="none", ecolor="black", elinewidth=1.3, capsize=5, zorder=4)
    ref_auc = vals[conds.index("ref")]
    ax2.axhline(ref_auc, color=COND_STYLE["ref"]["color"], linestyle=":", linewidth=1.3, zorder=1)
    for xi, c, v, h in zip(x, conds, vals, hi):
        ax2.text(xi, h + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        if c != "A":
            g = float(auc_tab.loc[auc_tab.condition == c, "gap_closure"].iloc[0]) * 100
            glo = float(auc_tab.loc[auc_tab.condition == c, "gap_ci_lo"].iloc[0]) * 100
            ghi = float(auc_tab.loc[auc_tab.condition == c, "gap_ci_hi"].iloc[0]) * 100
            txt = f"{g:.0f}%\n[{glo:.0f}, {ghi:.0f}]" if c != "ref" else "100%\n(by def.)"
            ax2.text(xi, v / 2, txt, ha="center", va="center",
                     fontsize=8.5, color="white", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["A", "A_joint-\nmask", "A_flip-\ncons", "B", "ref"])
    ax2.set_ylabel("Normalized label-efficiency AUC")
    ax2.set_title("AUC (95% seed-bootstrap CI) and A-to-ref gap closure")
    ax2.set_ylim(0, 0.80)
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, fontweight="bold", va="top")
    fig.tight_layout()
    save_fig(plt, fig, "fig_strengthened_suite")


# --------------------------------------------------------------------------- #
# analysis 2: E1 same-frame label-source factorial
# --------------------------------------------------------------------------- #
def analysis_e1(e1, cv):
    ref_rows = cv[cv.condition == "ref"][["seed", "round", "budget", "cam", "map50_95"]].assign(arm="gt")
    a_rows = cv[cv.condition == "A"][["seed", "round", "budget", "cam", "map50_95"]].assign(arm="A_adaptive")
    allr = pd.concat([e1[["arm", "seed", "round", "budget", "cam", "map50_95"]], ref_rows, a_rows], ignore_index=True)
    allr = allr[allr["round"] >= 1].copy()
    allr["budget"] = allr.budget.astype(float).round(2)
    tab = ms(allr, ["cam", "arm", "budget"])
    diffs = []
    for cam in ("c17", "c18"):
        sub = allr[allr.cam == cam]
        p = sub.pivot_table(index=["seed", "budget"], columns="arm", values="map50_95")
        for arm in ["A_adaptive", "gt_sigma20", "pseudo_A", "pseudo_init"]:
            d = (p[arm] - p["gt"]).dropna()
            for b in sorted(d.index.get_level_values("budget").unique()):
                v = d.xs(b, level="budget").values
                lo, hi = boot_ci(v)
                diffs.append(dict(cam=cam, arm=arm, budget=b, diff_vs_gt=v.mean(), ci_lo=lo, ci_hi=hi, n=len(v)))
    return {"tab": tab, "diffs": pd.DataFrame(diffs), "all": allr}


def fig_sameframe_e1(plt, res):
    tab, diffs = res["tab"], res["diffs"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)
    for k, (ax, cam) in enumerate(zip(axes, ("c17", "c18"))):
        for arm in E1_ARMS + ["A_adaptive"]:
            t = tab[(tab.cam == cam) & (tab.arm == arm)].sort_values("budget")
            st = E1_STYLE[arm]
            ax.errorbar(t.budget, t["mean"], yerr=t["std"], capsize=3, linewidth=2.0 if arm != "A_adaptive" else 1.6,
                        markersize=6, label=E1_LABEL[arm], zorder=3 if arm != "A_adaptive" else 2, **st)
        # paired difference vs GT at budget 0.40, as a text block (avoids overlapping labels)
        short = {"gt_sigma20": "GT + 20 px noise", "pseudo_init": "pseudo (frozen init)",
                 "pseudo_A": "pseudo (A teacher)", "A_adaptive": "original A (own frames)"}
        lines = ["Δ vs GT at budget 0.40 [95% CI]:"]
        for arm in ["gt_sigma20", "pseudo_init", "pseudo_A", "A_adaptive"]:
            r = diffs[(diffs.cam == cam) & (diffs.arm == arm) & (diffs.budget == 0.4)].iloc[0]
            lines.append(f"{short[arm]}: {r.diff_vs_gt:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        if cam == "c17":
            ax.set_ylim(0.0, 1.02); pos = dict(x=0.03, y=0.97, ha="left", va="top")
        else:
            ax.set_ylim(0.0, 0.44); pos = dict(x=0.03, y=0.97, ha="left", va="top")
        ax.text(pos["x"], pos["y"], "\n".join(lines), transform=ax.transAxes, ha=pos["ha"], va=pos["va"], fontsize=8,
                family="DejaVu Sans Mono", bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95), zorder=5)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.4]); ax.set_xticklabels(["0.10", "0.20", "0.40"]); ax.minorticks_off()
        ax.set_xlim(0.085, 0.47)
        ax.set_xlabel("Label budget (fraction of training pool)")
        ax.set_ylabel("mAP@[.50:.95]")
        ax.set_title(f"({'ab'[k]}) {'Camera17 (training view)' if cam == 'c17' else 'Camera18 (unseen view)'}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, framealpha=1.0, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Same-frame label-source factorial: frames fixed to ref's per-round lists, only the labels differ", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig(plt, fig, "fig_sameframe_e1")


# --------------------------------------------------------------------------- #
# analysis 3: cross-view
# --------------------------------------------------------------------------- #
def analysis_crossview(cv):
    d = cv.copy()
    tab = ms(d, ["cam", "condition", "budget"])
    aucs = per_seed_auc(d, ["cam", "condition"])
    auc_ms = aucs.groupby(["cam", "condition"]).auc.agg(["mean", "std"]).reset_index()
    summ = []
    for cam in sorted(d.cam.unique()):
        a = aucs[aucs.cam == cam].pivot(index="seed", columns="condition", values="auc")
        gap = (a.B - a.A) / (a.ref - a.A)
        rng = np.random.default_rng(0)                      # sequential RNG exactly as analyze_crossview.py
        glo, ghi = boot_ci(gap.values, rng=rng)
        row = dict(cam=cam, auc_A=a.A.mean(), auc_B=a.B.mean(), auc_ref=a.ref.mean(),
                   gap_closure=gap.mean(), gap_ci_lo=glo, gap_ci_hi=ghi)
        for c1, c2 in (("B", "A"), ("ref", "A"), ("B", "ref")):
            diff = a[c1] - a[c2]
            lo, hi = boot_ci(diff.values, rng=rng)
            row[f"d_{c1}_{c2}"] = diff.mean(); row[f"d_{c1}_{c2}_lo"] = lo; row[f"d_{c1}_{c2}_hi"] = hi
        summ.append(row)
    p = d.pivot_table(index=["condition", "seed", "budget"], columns="cam", values="map50_95").reset_index()
    p["drop"] = p.c17 - p.c18; p["rel"] = p["drop"] / p.c17
    drop = p.groupby(["condition", "budget"]).agg(c17=("c17", "mean"), c18=("c18", "mean"),
                                                  drop=("drop", "mean"), rel=("rel", "mean"),
                                                  rel_sd=("rel", "std")).reset_index()
    return {"tab": tab, "aucs": aucs, "auc_ms": auc_ms, "summ": pd.DataFrame(summ), "drop": drop}


def fig_crossview(plt, res):
    tab, drop, summ = res["tab"], res["drop"], res["summ"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for k, cam in enumerate(("c17", "c18")):
        ax = axes[k]
        for c in ("A", "B", "ref"):
            t = tab[(tab.cam == cam) & (tab.condition == c)].sort_values("budget")
            ax.errorbar(t.budget, t["mean"], yerr=t["std"], capsize=3, linewidth=2.2 if c == "B" else 1.9,
                        markersize=6, label=COND_LABEL[c], **COND_STYLE[c])
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels([f"{b:.2f}" for b in BUDGETS]); ax.minorticks_off()
        ax.set_xlabel("Label budget (fraction of training pool)")
        ax.set_ylabel("mAP@[.50:.95]")
        s = summ[summ.cam == cam].iloc[0]
        ax.set_title(f"({'ab'[k]}) {'Camera17 (training view)' if cam == 'c17' else 'Camera18 (unseen view)'}")
        ax.text(0.03, 0.97, f"B gap closure {s.gap_closure*100:.0f}% [{s.gap_ci_lo*100:.0f}, {s.gap_ci_hi*100:.0f}]\n"
                            f"AUC B−ref {s.d_B_ref:+.3f} [{s.d_B_ref_lo:+.3f}, {s.d_B_ref_hi:+.3f}]",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95))
        ax.set_ylim(0, 0.9)
    axes[0].legend(loc="lower left", framealpha=1.0)
    ax = axes[2]
    conds = ["A", "B", "ref"]; w = 0.26; x = np.arange(len(BUDGETS))
    for i, c in enumerate(conds):
        t = drop[drop.condition == c].sort_values("budget")
        ax.bar(x + (i - 1) * w, t.rel * 100, width=w, yerr=t.rel_sd * 100, capsize=3,
               color=COND_STYLE[c]["color"], edgecolor="black", linewidth=0.6, label=c, zorder=3)
        for xi, v in zip(x + (i - 1) * w, t.rel * 100):
            ax.text(xi, v + 4.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{b:.2f}" for b in BUDGETS])
    ax.set_xlabel("Label budget (fraction of training pool)")
    ax.set_ylabel("Relative mAP@[.50:.95] drop, c17 → c18 (%)")
    ax.set_title("(c) Cross-view drop per checkpoint (mean ± SD)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", framealpha=1.0, title="condition")
    fig.tight_layout()
    save_fig(plt, fig, "fig_crossview")


# --------------------------------------------------------------------------- #
# analysis 4: downstream kinematics c17 vs c18
# --------------------------------------------------------------------------- #
def analysis_downstream(d17, d18, md17, md18):
    """Both cameras from per-seed JSON. SD with ddof=0 (population SD) to match the convention of the
    server's analyze_downstream_eval.py tables, against which the means/SDs are cross-checked."""
    metrics = ["joint_angle_mae", "rom_error", "jerk_error"]
    d = pd.concat([d17.assign(cam="c17"), d18.assign(cam="c18")], ignore_index=True)
    long = d.melt(id_vars=["cam", "cond", "seed", "budget"], value_vars=metrics, var_name="metric")
    tab = long.groupby(["cam", "cond", "budget", "metric"]).value.agg(mean="mean", std=lambda v: v.std(ddof=0)).reset_index()
    chk = pd.concat([md17.assign(cam="c17"), md18.assign(cam="c18")], ignore_index=True).merge(
        tab, on=["cam", "cond", "budget", "metric"], suffixes=("_md", "_json"))
    # seed-paired contrasts at budget 0.40, both cameras
    rows = []
    for cam in ("c17", "c18"):
        p = d[(d.cam == cam) & (d.budget == 0.4)].pivot(index="seed", columns="cond", values=metrics)
        for m in metrics:
            for c1, c2 in (("B", "A"), ("B", "ref"), ("ref", "A")):
                v = (p[(m, c1)] - p[(m, c2)]).values
                lo, hi = boot_ci(v)
                rows.append(dict(cam=cam, metric=m, contrast=f"{c1}−{c2}", diff=v.mean(), ci_lo=lo, ci_hi=hi, n=len(v)))
    # paired c17 -> c18 change per checkpoint at 0.40 (same seed, same weights)
    for m in metrics:
        p = d[d.budget == 0.4].pivot_table(index=["cond", "seed"], columns="cam", values=m)
        for c in ("A", "B", "ref"):
            v = (p.loc[c, "c18"] - p.loc[c, "c17"]).values
            lo, hi = boot_ci(v)
            rows.append(dict(cam="c18−c17", metric=m, contrast=c, diff=v.mean(), ci_lo=lo, ci_hi=hi, n=len(v)))
    return {"raw": d, "tab": tab, "contrasts": pd.DataFrame(rows), "chk": chk}


def fig_downstream(plt, res):
    tab = res["tab"]
    panels = [("rom_error", "ROM error (deg, lower = better)", "(a) Range-of-motion error"),
              ("joint_angle_mae", "Joint-angle MAE (deg, lower = better)", "(b) Joint-angle MAE")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    conds = ["A", "B", "ref"]; cams = ["c17", "c18"]
    x = np.arange(len(BUDGETS)); n = len(conds) * len(cams); w = 0.8 / n
    for ax, (m, ylab, title) in zip(axes, panels):
        for i, c in enumerate(conds):
            for j, cam in enumerate(cams):
                t = tab[(tab.cam == cam) & (tab.cond == c) & (tab.metric == m)].sort_values("budget")
                off = (i * 2 + j - (n - 1) / 2) * w
                kw = dict(color=COND_STYLE[c]["color"], edgecolor="black", linewidth=0.6, zorder=3)
                if cam == "c18":
                    kw.update(facecolor="white", edgecolor=COND_STYLE[c]["color"], hatch="///", linewidth=1.2)
                ax.bar(x + off, t["mean"], width=w, yerr=t["std"], capsize=2,
                       error_kw=dict(elinewidth=0.9), label=f"{c} ({'Camera17' if cam == 'c17' else 'Camera18'})", **kw)
        ax.set_xticks(x); ax.set_xticklabels([f"{b:.2f}" for b in BUDGETS])
        ax.set_xlabel("Label budget (fraction of training pool)")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, framealpha=1.0, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Downstream kinematic fidelity, YOLO11n: filled = Camera17 (training view), hatched = Camera18 (unseen view); mean ± SD over 6 seeds (population SD)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(plt, fig, "fig_downstream_c17_c18")


# --------------------------------------------------------------------------- #
# supplementary: inference-resolution sensitivity
# --------------------------------------------------------------------------- #
def analysis_resolution(cv416, cv640, diag):
    a = cv416.assign(imgsz=416); b = cv640.assign(imgsz=640)
    d = pd.concat([a, b], ignore_index=True)
    tab = d.groupby(["imgsz", "cam", "condition", "budget"]).map50_95.agg(["mean", "std"]).reset_index()
    return {"tab": tab, "diag": diag}


def fig_resolution(plt, res):
    tab, diag = res["tab"], res["diag"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    nr = diag[~diag.rect].sort_values("imgsz"); r = diag[diag.rect].sort_values("imgsz")
    ax.plot(nr.imgsz, nr.pose_map50_95, "-o", color="#009E73", label="pose mAP@[.50:.95] (rect=False)")
    ax.plot(nr.imgsz, nr.box_map50, "--s", color="#555555", label="box mAP@50 (rect=False)")
    ax.plot(r.imgsz, r.pose_map50_95, "^", color="#009E73", markersize=9, markerfacecolor="white", label="pose mAP@[.50:.95] (rect=True)")
    ax.plot(r.imgsz, r.box_map50, "v", color="#555555", markersize=9, markerfacecolor="white", label="box mAP@50 (rect=True)")
    ax.axvline(416, color="0.6", linestyle=":", linewidth=1.2)
    ax.text(424, 0.70, "training\nresolution\n(416 px)", ha="left", va="center", fontsize=8.5, color="0.4")
    for xi, yi in zip(nr.imgsz, nr.pose_map50_95):
        ax.text(xi, yi + 0.03, f"{yi:.3f}", ha="center", fontsize=8.5, color="#009E73")
    ax.set_xticks(sorted(diag.imgsz.unique()))
    ax.set_xlabel("Inference image size (px)"); ax.set_ylabel("Metric on Camera18")
    ax.set_title("(a) ref, seed 0, budget 0.40, Camera18")
    ax.set_ylim(-0.03, 1.12); ax.legend(loc="center right", fontsize=8.5, framealpha=1.0)
    for k, cam in enumerate(("c17", "c18")):
        ax = axes[k + 1]
        for c in ("A", "B", "ref"):
            for imgsz, alpha, lsz in ((416, 1.0, "416 px"), (640, 0.45, "640 px")):
                t = tab[(tab.cam == cam) & (tab.condition == c) & (tab.imgsz == imgsz)].sort_values("budget")
                st = dict(COND_STYLE[c]); st["alpha"] = alpha
                if imgsz == 640:
                    st["markerfacecolor"] = "white"
                ax.errorbar(t.budget, t["mean"], yerr=t["std"], capsize=3, linewidth=1.9, markersize=6,
                            label=f"{c}, {lsz}", **st)
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels([f"{b:.2f}" for b in BUDGETS]); ax.minorticks_off()
        ax.set_xlabel("Label budget (fraction of training pool)"); ax.set_ylabel("mAP@[.50:.95]")
        ax.set_title(f"({'bc'[k]}) {'Camera17' if cam == 'c17' else 'Camera18'}: inference at 416 vs 640 px")
        ax.set_ylim(0, 0.9)
        ax.legend(loc="upper left", ncol=2, fontsize=8.5, framealpha=1.0)
    fig.suptitle("Supplementary: inference-resolution sensitivity of models trained at 416 px (not a fairness comparison)", fontsize=11)
    fig.tight_layout()
    save_fig(plt, fig, "fig_resolution_sensitivity")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def write_tables(S, E, C, D, R, suite, grid, conf):
    md = []
    # ---- T1
    conds = ["A", "A_jointmask", "A_flipcons", "B", "ref"]
    rows = []
    for c in conds:
        t = S["tab"][S["tab"].condition == c].set_index("budget")
        a = S["auc_tab"][S["auc_tab"].condition == c].iloc[0]
        row = {"condition": c}
        for b in BUDGETS:
            row[f"mAP50-95 @{b:.2f}"] = fmt_pm(t.loc[b, "mean"], t.loc[b, "std"])
        row["AUC (mean ± SD)"] = fmt_pm(a.auc_mean, a.auc_sd)
        row["AUC 95% CI"] = f"[{a.auc_ci_lo:.3f}, {a.auc_ci_hi:.3f}]"
        row["ΔAUC vs A [95% CI]"] = "—" if c == "A" else f"{a.dauc_vs_A:+.3f} [{a.dauc_ci_lo:+.3f}, {a.dauc_ci_hi:+.3f}]"
        row["Gap closure % [95% CI]"] = "0 (reference)" if c == "A" else f"{a.gap_closure*100:.1f} [{a.gap_ci_lo*100:.1f}, {a.gap_ci_hi*100:.1f}]"
        rows.append(row)
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "table_T1_strengthened_suite.csv", index=False, encoding="utf-8-sig")
    S["tab"].to_csv(OUT / "data_fig_strengthened_suite_curves.csv", index=False)
    S["auc_tab"].to_csv(OUT / "data_fig_strengthened_suite_auc.csv", index=False)
    S["aucs"].to_csv(OUT / "data_strengthened_suite_auc_per_seed.csv", index=False)
    md.append("## Table T1. Strengthened GT-free self-training arms vs the main grid (Camera17, YOLO11n, 6 seeds)\n\n"
              "mAP@[.50:.95] per budget (mean ± SD over seeds; budget 0.05 is the shared 5% GT initialization), "
              "normalized label-efficiency AUC, ΔAUC vs A, and A-to-ref gap closure (mean over seeds of per-seed ratio). "
              "95% CI = seed-bootstrap (10,000 resamples) of the mean.\n\n" + md_table(t1))

    # ---- T2
    arms = ["gt", "gt_sigma20", "pseudo_init", "pseudo_A", "A_adaptive"]
    rows = []
    for cam in ("c17", "c18"):
        for arm in arms:
            t = E["tab"][(E["tab"].cam == cam) & (E["tab"].arm == arm)].set_index("budget")
            row = {"camera": cam, "label source": E1_LABEL[arm]}
            for b in (0.1, 0.2, 0.4):
                row[f"mAP50-95 @{b:.2f}"] = fmt_pm(t.loc[b, "mean"], t.loc[b, "std"])
            for b in (0.1, 0.2, 0.4):
                if arm == "gt":
                    row[f"Δ vs GT @{b:.2f} [95% CI]"] = "—"
                else:
                    r = E["diffs"][(E["diffs"].cam == cam) & (E["diffs"].arm == arm) & (E["diffs"].budget == b)].iloc[0]
                    row[f"Δ vs GT @{b:.2f} [95% CI]"] = f"{r.diff_vs_gt:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
            rows.append(row)
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "table_T2_sameframe_e1.csv", index=False, encoding="utf-8-sig")
    E["tab"].to_csv(OUT / "data_fig_sameframe_e1_curves.csv", index=False)
    E["diffs"].to_csv(OUT / "data_fig_sameframe_e1_paired_diffs.csv", index=False)
    md.append("## Table T2. Same-frame label-source factorial (E1): frames fixed to ref's per-round lists, labels swapped\n\n"
              "mAP@[.50:.95] (mean ± SD over 6 seeds) and seed-paired difference vs GT labels on the same frames "
              "[95% seed-bootstrap CI]. 'original A' trains on its own adaptively selected frames and is shown for reference only.\n\n"
              + md_table(t2))

    # ---- T3
    rows = []
    for cam in ("c17", "c18"):
        s = C["summ"][C["summ"].cam == cam].iloc[0]
        for c in ("A", "B", "ref"):
            t = C["tab"][(C["tab"].cam == cam) & (C["tab"].condition == c)].set_index("budget")
            a = C["auc_ms"][(C["auc_ms"].cam == cam) & (C["auc_ms"].condition == c)].iloc[0]
            row = {"camera": cam, "condition": c}
            for b in BUDGETS:
                row[f"mAP50-95 @{b:.2f}"] = fmt_pm(t.loc[b, "mean"], t.loc[b, "std"])
            row["AUC (mean ± SD)"] = fmt_pm(a["mean"], a["std"])
            row["ΔAUC vs A [95% CI]"] = ("—" if c == "A" else
                                         f"{s[f'd_{c}_A']:+.3f} [{s[f'd_{c}_A_lo']:+.3f}, {s[f'd_{c}_A_hi']:+.3f}]")
            row["ΔAUC B−ref [95% CI]"] = (f"{s.d_B_ref:+.3f} [{s.d_B_ref_lo:+.3f}, {s.d_B_ref_hi:+.3f}]" if c == "B" else "")
            row["B gap closure % [95% CI]"] = (f"{s.gap_closure*100:.1f} [{s.gap_ci_lo*100:.1f}, {s.gap_ci_hi*100:.1f}]" if c == "B" else "")
            rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "table_T3_crossview.csv", index=False, encoding="utf-8-sig")
    C["tab"].to_csv(OUT / "data_fig_crossview_curves.csv", index=False)
    C["drop"].to_csv(OUT / "data_fig_crossview_relative_drop.csv", index=False)
    C["summ"].to_csv(OUT / "data_crossview_auc_summary.csv", index=False)
    C["aucs"].to_csv(OUT / "data_crossview_auc_per_seed.csv", index=False)
    md.append("## Table T3. Cross-view evaluation of Camera17-trained checkpoints on Camera17 and unseen Camera18 (YOLO11n, 6 seeds, 416 px)\n\n"
              "mAP@[.50:.95] per budget (mean ± SD), normalized AUC, AUC contrasts and B gap closure [95% seed-bootstrap CI].\n\n"
              + md_table(t3))
    dr = C["drop"].copy(); dr["rel"] = dr.rel * 100; dr["rel_sd"] = dr.rel_sd * 100
    dr = dr.rename(columns={"c17": "mAP c17", "c18": "mAP c18", "drop": "absolute drop", "rel": "relative drop % (mean)", "rel_sd": "relative drop % (SD)"})
    dr["budget"] = dr.budget.map(lambda b: f"{b:.2f}")
    for c in ("relative drop % (mean)", "relative drop % (SD)"):
        dr[c] = dr[c].map(lambda v: f"{v:.1f}")
    md.append("### Table T3b. Paired c17 → c18 drop per checkpoint (mean over seeds)\n\n" + md_table(dr))

    # ---- T4
    metrics = [("joint_angle_mae", "Joint-angle MAE (deg)"), ("rom_error", "ROM error (deg)"), ("jerk_error", "Jerk-magnitude error (diag·s⁻³)")]
    rows = []
    for m, lab in metrics:
        row = {"metric (budget 0.40)": lab}
        nd = 2 if m != "jerk_error" else 3
        for c in ("A", "B", "ref"):
            for cam in ("c17", "c18"):
                t = D["tab"][(D["tab"].cam == cam) & (D["tab"].cond == c) & (D["tab"].metric == m) & (D["tab"].budget == 0.4)].iloc[0]
                row[f"{c} {cam}"] = fmt_pm(t["mean"], t["std"], nd)
        for ct in ("B−A", "B−ref"):
            for cam in ("c17", "c18"):
                r = D["contrasts"][(D["contrasts"].cam == cam) & (D["contrasts"].metric == m) & (D["contrasts"].contrast == ct)].iloc[0]
                row[f"{ct} {cam} [95% CI]"] = f"{r['diff']:+.{nd}f} [{r.ci_lo:+.{nd}f}, {r.ci_hi:+.{nd}f}]"
        rows.append(row)
    t4 = pd.DataFrame(rows)
    t4.to_csv(OUT / "table_T4_downstream_c17_c18_b040.csv", index=False, encoding="utf-8-sig")
    D["tab"].to_csv(OUT / "data_fig_downstream_c17_c18.csv", index=False)
    D["contrasts"].to_csv(OUT / "data_downstream_contrasts_b040.csv", index=False)
    D["raw"].to_csv(OUT / "data_downstream_per_seed.csv", index=False)
    md.append("## Table T4. Downstream kinematic fidelity at budget 0.40, Camera17 vs unseen Camera18 (YOLO11n, mean ± SD over 6 seeds, population SD; lower = better)\n\n"
              "Camera17 from the corrected 2026-08-09 downstream run (`downstream_11n_u30_c17_corrected_2026-08-09/`), Camera18 from "
              "`downstream_11n_u30_c18/`; both evaluated with the same post-correction code. Contrasts are seed-paired mean differences "
              "with 95% seed-bootstrap CI; negative B−A favours B.\n\n" + md_table(t4))
    # T4b: paired c17 -> c18 change per condition at 0.40
    rows = []
    for m, lab in metrics:
        nd = 2 if m != "jerk_error" else 3
        row = {"metric (budget 0.40)": lab}
        for c in ("A", "B", "ref"):
            r = D["contrasts"][(D["contrasts"].cam == "c18−c17") & (D["contrasts"].metric == m) & (D["contrasts"].contrast == c)].iloc[0]
            row[f"{c}: c18 − c17 [95% CI]"] = f"{r['diff']:+.{nd}f} [{r.ci_lo:+.{nd}f}, {r.ci_hi:+.{nd}f}]"
        rows.append(row)
    md.append("### Table T4b. Paired Camera17 → Camera18 change of each checkpoint at budget 0.40 (same seed, same weights; mean [95% seed-bootstrap CI]; positive = worse on Camera18)\n\n"
              + md_table(pd.DataFrame(rows)))

    # ---- T5
    rej = suite.groupby(["condition", "budget"])[["n_proposed", "n_accepted", "n_rejected"]].agg(["mean", "std", "min", "max"]).reset_index()
    rej.columns = ["condition", "budget"] + [f"{a}_{b}" for a, b in rej.columns[2:]]
    rej["reject_rate_of_proposed"] = rej.n_rejected_mean / rej.n_proposed_mean.replace(0, np.nan)
    rej.to_csv(OUT / "table_T5_rejection_accounting.csv", index=False, encoding="utf-8-sig")
    rows = []
    for _, r in rej.iterrows():
        rows.append({"arm": r.condition, "budget": f"{r.budget:.2f}",
                     "n_proposed (mean)": f"{r.n_proposed_mean:.0f}",
                     "n_accepted (mean)": f"{r.n_accepted_mean:.0f}",
                     "n_rejected (mean ± SD, min–max)": f"{r.n_rejected_mean:.1f} ± {r.n_rejected_std:.1f} ({r.n_rejected_min:.0f}–{r.n_rejected_max:.0f})",
                     "rejected / proposed": "—" if np.isnan(r.reject_rate_of_proposed) else f"{r.reject_rate_of_proposed*100:.1f}%"})
    # comparison rows from the original package (n_proposed not recorded there)
    cmp = []
    if conf is not None:
        cc = conf.groupby(["condition", "budget"])[["n_accepted", "n_rejected"]].agg(["mean", "std", "min", "max"]).reset_index()
        cc.columns = ["condition", "budget"] + [f"{a}_{b}" for a, b in cc.columns[2:]]
        cmp.append(cc)
    gb = grid[grid.condition == "B"].groupby(["condition", "budget"])[["n_accepted", "n_rejected"]].agg(["mean", "std", "min", "max"]).reset_index()
    gb.columns = ["condition", "budget"] + [f"{a}_{b}" for a, b in gb.columns[2:]]
    cmp.append(gb)
    cmp = pd.concat(cmp, ignore_index=True)
    cmp.to_csv(OUT / "table_T5b_rejection_original_package.csv", index=False, encoding="utf-8-sig")
    for _, r in cmp.iterrows():
        rows.append({"arm": f"{r.condition} (original package)", "budget": f"{r.budget:.2f}",
                     "n_proposed (mean)": "n/r", "n_accepted (mean)": f"{r.n_accepted_mean:.0f}",
                     "n_rejected (mean ± SD, min–max)": f"{r.n_rejected_mean:.1f} ± {r.n_rejected_std:.1f} ({r.n_rejected_min:.0f}–{r.n_rejected_max:.0f})",
                     "rejected / proposed": "n/r"})
    t5 = pd.DataFrame(rows)
    md.append("## Table T5. Proposal/acceptance/rejection accounting per round (mean over 6 seeds)\n\n"
              "Strengthened arms from experiment_curve.csv; comparison rows (A_conf from clinician_i3_conf, B σ=0 from clinician_i4_uniform30) "
              "from the original code package, where n_proposed was not recorded (n/r). Round 0 (budget 0.05) is the shared GT initialization with no proposals.\n\n"
              + md_table(t5))

    (OUT / "tables.md").write_text("# Tables (generated by make_figures.py)\n\n" + "\n\n".join(md) + "\n", encoding="utf-8")
    print("  wrote tables.md + table_T*.csv")

    # resolution data table
    R["tab"].to_csv(OUT / "data_fig_resolution_sensitivity_curves.csv", index=False)
    R["diag"].to_csv(OUT / "data_fig_resolution_sensitivity_diag.csv", index=False)


# --------------------------------------------------------------------------- #
# sanity check against the reference text outputs
# --------------------------------------------------------------------------- #
def sanity_check(S, E, C, D):
    lines = []; worst = 0.0; nfail = 0

    def cmp(name, mine, ref, tol=0.001):
        nonlocal worst, nfail
        mine = float(mine); ref = float(ref); d = abs(mine - ref); worst = max(worst, d)
        ok = d <= tol + 1e-12
        if not ok:
            nfail += 1
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name:70s} mine={mine:+.4f} ref={ref:+.4f} |d|={d:.4f}")

    # ---- analysis_output_final.txt (suite + E1)
    txt = INPUTS["e1_ref_txt"].read_text(encoding="utf-8")
    # suite per-budget means
    sec1 = txt.split("====================")[1]
    for line in sec1.splitlines():
        m = re.match(r"^(A|A_flipcons|A_jointmask|B|ref)\s+((?:[0-9.]+\s+){12})", line)
        if m:
            vals = [float(v) for v in m.group(2).split()]
            t = S["tab"][S["tab"].condition == m.group(1)].set_index("budget")
            for b, v in zip(BUDGETS, vals[:4]):
                cmp(f"suite mean {m.group(1)} @{b:.2f}", t.loc[b, "mean"], v)
            for b, v in zip(BUDGETS, vals[4:8]):
                cmp(f"suite std {m.group(1)} @{b:.2f}", t.loc[b, "std"], v)
    for m in re.finditer(r"^\s*(A_jointmask|A_flipcons|B|ref)\s+- A: ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]\s+gap closure\s+([0-9.]+)%", sec1, re.M):
        a = S["auc_tab"][S["auc_tab"].condition == m.group(1)].iloc[0]
        cmp(f"suite dAUC {m.group(1)}-A", a.dauc_vs_A, m.group(2))
        cmp(f"suite dAUC CI lo {m.group(1)}", a.dauc_ci_lo, m.group(3))
        cmp(f"suite dAUC CI hi {m.group(1)}", a.dauc_ci_hi, m.group(4))
        cmp(f"suite gap closure % {m.group(1)}", a.gap_closure * 100, m.group(5), tol=0.1)
    for m in re.finditer(r"^(A|A_flipcons|A_jointmask|B|ref)\s+([0-9.]+)\s+([0-9.]+)\s*$", sec1, re.M):
        a = S["auc_tab"][S["auc_tab"].condition == m.group(1)].iloc[0]
        cmp(f"suite AUC mean {m.group(1)}", a.auc_mean, m.group(2))
        cmp(f"suite AUC sd {m.group(1)}", a.auc_sd, m.group(3))
    # E1
    sec2 = txt.split("====================")[2]
    name_map = {"A adaptive (own frames)": "A_adaptive", "gt (=ref, same frames)": "gt", "gt_sigma20": "gt_sigma20",
                "pseudo_A": "pseudo_A", "pseudo_init": "pseudo_init"}
    for cam_block in re.split(r"-- (c1[78]):", sec2)[1:][::1]:
        pass
    parts = re.split(r"-- (c1[78]):", sec2)
    for cam, block in zip(parts[1::2], parts[2::2]):
        for line in block.splitlines():
            m = re.match(r"^(A adaptive \(own frames\)|gt \(=ref, same frames\)|gt_sigma20|pseudo_A|pseudo_init)\s+((?:[0-9.]+\s+){9})", line)
            if m:
                arm = name_map[m.group(1)]; vals = [float(v) for v in m.group(2).split()]
                t = E["tab"][(E["tab"].cam == cam) & (E["tab"].arm == arm)].set_index("budget")
                for b, v in zip((0.1, 0.2, 0.4), vals[:3]):
                    cmp(f"E1 {cam} mean {arm} @{b:.2f}", t.loc[b, "mean"], v)
                for b, v in zip((0.1, 0.2, 0.4), vals[3:6]):
                    cmp(f"E1 {cam} std {arm} @{b:.2f}", t.loc[b, "std"], v)
        for m in re.finditer(r"^\s+(A adaptive \(own frames\)|gt_sigma20|pseudo_A|pseudo_init)\s+budget ([0-9.]+): ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", block, re.M):
            arm = name_map[m.group(1)]; b = float(m.group(2))
            r = E["diffs"][(E["diffs"].cam == cam) & (E["diffs"].arm == arm) & (E["diffs"].budget == b)].iloc[0]
            cmp(f"E1 {cam} diff {arm} @{b:.2f}", r.diff_vs_gt, m.group(3))
            cmp(f"E1 {cam} diff CI lo {arm} @{b:.2f}", r.ci_lo, m.group(4))
            cmp(f"E1 {cam} diff CI hi {arm} @{b:.2f}", r.ci_hi, m.group(5))

    # ---- crossview_analysis_output.txt
    txt = INPUTS["cv_ref_txt"].read_text(encoding="utf-8")
    cam = cond = None
    for line in txt.split("== normalized AUC")[0].splitlines():
        m = re.match(r"^(c1[78])?\s*(A|B|ref)?\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", line)
        if m:
            cam = m.group(1) or cam; cond = m.group(2) or cond
            t = C["tab"][(C["tab"].cam == cam) & (C["tab"].condition == cond) & (np.isclose(C["tab"].budget, float(m.group(3))))].iloc[0]
            cmp(f"CV {cam} {cond} mean @{float(m.group(3)):.2f}", t["mean"], m.group(4))
            cmp(f"CV {cam} {cond} std @{float(m.group(3)):.2f}", t["std"], m.group(5))
    blk = txt.split("== normalized AUC")[1].split("== c17")[0]
    cam = None
    for line in blk.splitlines():
        m = re.match(r"^(c1[78])?\s*(A|B|ref)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if m:
            cam = m.group(1) or cam
            a = C["auc_ms"][(C["auc_ms"].cam == cam) & (C["auc_ms"].condition == m.group(2))].iloc[0]
            cmp(f"CV AUC mean {cam} {m.group(2)}", a["mean"], m.group(3))
            cmp(f"CV AUC sd {cam} {m.group(2)}", a["std"], m.group(4))
    for m in re.finditer(r"== (c1[78]): B gap closure mean=([0-9.]+)\s+95% seed-bootstrap CI=\[([0-9.]+), ([0-9.]+)\]", txt):
        s = C["summ"][C["summ"].cam == m.group(1)].iloc[0]
        cmp(f"CV gap closure {m.group(1)}", s.gap_closure, m.group(2))
        cmp(f"CV gap closure CI lo {m.group(1)}", s.gap_ci_lo, m.group(3))
        cmp(f"CV gap closure CI hi {m.group(1)}", s.gap_ci_hi, m.group(4))
    for camblk in re.split(r"== (c1[78]):", txt)[1:]:
        pass
    parts = re.split(r"== (c1[78]): B gap", txt)
    for cam, block in zip(parts[1::2], parts[2::2]):
        s = C["summ"][C["summ"].cam == cam].iloc[0]
        for m in re.finditer(r"AUC (B|ref)-(A|ref): ([+-][0-9.]+) \[([+-][0-9.]+), ([+-][0-9.]+)\]", block.split("== paired")[0]):
            k = f"d_{m.group(1)}_{m.group(2)}"
            cmp(f"CV {cam} AUC {m.group(1)}-{m.group(2)}", s[k], m.group(3))
            cmp(f"CV {cam} AUC {m.group(1)}-{m.group(2)} CI lo", s[k + "_lo"], m.group(4))
            cmp(f"CV {cam} AUC {m.group(1)}-{m.group(2)} CI hi", s[k + "_hi"], m.group(5))
    cond = None
    for line in txt.split("== paired c17->c18 drop")[1].splitlines():
        m = re.match(r"^(A|B|ref)?\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if m:
            cond = m.group(1) or cond
            r = C["drop"][(C["drop"].condition == cond) & (np.isclose(C["drop"].budget, float(m.group(2))))].iloc[0]
            cmp(f"CV drop {cond} @{float(m.group(2)):.2f} rel", r.rel, m.group(6))
            cmp(f"CV drop {cond} @{float(m.group(2)):.2f} abs", r["drop"], m.group(5))

    # ---- downstream reference tables (c17 corrected 08-09 md, c18 md) vs per-seed JSON (mean and population SD)
    for _, r in D["chk"].iterrows():
        cmp(f"DS {r.cam} {r.metric} {r.cond} @{r.budget:.2f} mean (md vs json)", r.mean_json, r.mean_md)
        cmp(f"DS {r.cam} {r.metric} {r.cond} @{r.budget:.2f} sd   (md vs json)", r.std_json, r.std_md)

    head = (f"Sanity check of make_figures.py against reference outputs\n"
            f"  {INPUTS['e1_ref_txt']}\n  {INPUTS['cv_ref_txt']}\n"
            f"  {INPUTS['down_c17_md']} (vs per-seed JSON, corrected 2026-08-09 run)\n  {INPUTS['down_c18_md']} (vs per-seed JSON)\n"
            f"c17 downstream JSON source used: {INPUTS.get('_c17_source_used')}\n"
            f"comparisons: {len(lines)}   failures (|d| > 0.001; gap-closure % tol 0.1): {nfail}   max |d| = {worst:.4f}\n\n")
    (OUT / "sanity_check.txt").write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  sanity check: {len(lines)} comparisons, {nfail} failures, max |d| = {worst:.4f}")
    return nfail, len(lines), worst


# --------------------------------------------------------------------------- #
def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)

    print("loading inputs")
    suite = load_suite(); grid = load_grid()
    conf = pd.read_csv(INPUTS["conf_csv"]) if INPUTS["conf_csv"].exists() else None
    e1 = pd.read_csv(INPUTS["e1_csv"]); cv = pd.read_csv(INPUTS["cv_csv"]); cv640 = pd.read_csv(INPUTS["cv640_csv"])
    md17 = parse_downstream_md(INPUTS["down_c17_md"]); md18 = parse_downstream_md(INPUTS["down_c18_md"])
    try:
        d17 = load_downstream(INPUTS["down_c17_dir"]); c17_source = str(INPUTS["down_c17_dir"])
        assert len(d17) == 72
    except (OSError, AssertionError) as e:   # primary copy unreadable -> verified package copy
        print(f"  [note] corrected c17 JSONs not readable in the shared folder ({type(e).__name__}); "
              f"using the package copy {INPUTS['down_c17_dir_fallback']}")
        d17 = load_downstream(INPUTS["down_c17_dir_fallback"]); c17_source = str(INPUTS["down_c17_dir_fallback"]) + "  (fallback)"
    d18 = load_downstream(INPUTS["down_c18_dir"])
    assert len(d17) == 72 and len(d18) == 72, (len(d17), len(d18))
    INPUTS["_c17_source_used"] = c17_source
    diag = parse_diag640(INPUTS["diag640_txt"])
    assert len(cv) == 144 and cv.session.nunique() == 18
    assert len(e1) == 108 and e1.groupby(["arm", "seed", "round"]).ngroups == 54

    print("analysing")
    S = analysis_suite(suite, grid)
    E = analysis_e1(e1, cv)
    C = analysis_crossview(cv)
    D = analysis_downstream(d17, d18, md17, md18)
    R = analysis_resolution(cv, cv640, diag)

    print("figures")
    fig_strengthened_suite(plt, S)
    fig_sameframe_e1(plt, E)
    fig_crossview(plt, C)
    fig_downstream(plt, D)
    fig_resolution(plt, R)

    print("tables")
    write_tables(S, E, C, D, R, suite, grid, conf)
    nfail, n, worst = sanity_check(S, E, C, D)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
