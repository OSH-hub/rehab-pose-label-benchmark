"""Usage: KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 python scripts/plot_result_figures.py"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.metrics.learning_curve import area_under_learning_curve  # noqa: E402

EXP = _HERE.parent / "outputs" / "experiments"
OUTDIR = _HERE.parent / "verification" / "paper_figures"


def _curve_auc_per_seed(csv_path, condition, sigma=None, metric="map50_95"):
    per_seed = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] != condition:
                continue
            if sigma is not None and r.get("sigma", "") != sigma:
                continue
            per_seed[int(r["seed"])][float(r["budget"])] = float(r[metric])
    out = {}
    for seed, bd in per_seed.items():
        budgets = sorted(bd)
        scores = [bd[b] for b in budgets]
        out[seed] = area_under_learning_curve(budgets, scores, normalize=True)
    return out


def _auc_mean_std(csv_path, condition, sigma=None):
    vals = np.array(list(_curve_auc_per_seed(csv_path, condition, sigma).values()))
    return float(vals.mean()), float(vals.std(ddof=0))


def _load_downstream(ddir):
    METRICS = ["joint_angle_mae_mean", "rom_error_mean", "smoothness_error_mean"]
    data = {m: defaultdict(lambda: defaultdict(list)) for m in METRICS}
    budgets_seen = set()
    for f in sorted(Path(ddir).glob("*_downstream.json")):
        name = f.stem.replace("_downstream", "")
        if "_s" not in name:
            continue
        cond, _, seed = name.rpartition("_s")
        try:
            int(seed)
        except ValueError:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for _bk, bv in d.get("budgets", {}).items():
            if "error" in bv:
                raise ValueError(f"invalid downstream result {f.name}:{_bk}: {bv['error']}")
            b = float(bv["budget"])
            budgets_seen.add(b)
            agg = bv.get("aggregate", {})
            for m in METRICS:
                val = agg.get(m)
                if isinstance(val, dict):
                    val = val.get("mean")
                if val is not None:
                    data[m][cond][b].append(float(val))
    return data, sorted(budgets_seen)


def _ms(vals):
    arr = np.array([v for v in vals if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def fig_mechanism_decomp(plt, style):
    from _figstyle import save_fig

    i2 = EXP / "clinician_i2_decomp" / "clinician_curve.csv"
    i4 = EXP / "clinician_i4_uniform30" / "clinician_curve.csv"
    a_mu, _ = _auc_mean_std(i4, "A")
    filt_mu, _ = _auc_mean_std(i2, "A_filter")
    corr_mu, _ = _auc_mean_std(i2, "B_correct_only")
    b_mu, _ = _auc_mean_std(i2, "ref_filter")

    total = b_mu - a_mu
    filtering = filt_mu - a_mu
    correction = corr_mu - a_mu
    interaction = total - filtering - correction

    ci = {
        "total": (0.2972, 0.3358),
        "filtering": (0.0049, 0.0302),
        "correction": (0.3022, 0.3528),
        "interaction": (-0.0409, -0.0136),
    }

    labels = ["Total gap\n(B - A)", "Filtering\n(segmentation\nflags)",
              "Pose-label\nreplacement\n(correct only)", "Interaction"]
    vals = [total, filtering, correction, interaction]
    colors = ["#555555", style["A"]["color"], style["B"]["color"], "#999999"]
    err_lo = [vals[0] - ci["total"][0], vals[1] - ci["filtering"][0],
              vals[2] - ci["correction"][0], vals[3] - ci["interaction"][0]]
    err_hi = [ci["total"][1] - vals[0], ci["filtering"][1] - vals[1],
              ci["correction"][1] - vals[2], ci["interaction"][1] - vals[3]]

    fig, ax = plt.subplots(figsize=(7.2, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.62, edgecolor="black",
                  linewidth=0.6, zorder=3)
    ax.errorbar(x, vals, yerr=[err_lo, err_hi], fmt="none",
                ecolor="black", elinewidth=1.4, capsize=5, zorder=4)
    ax.axhline(0, color="black", linewidth=0.8)
    for xi, v, eh in zip(x, vals, err_hi):
        if v >= 0:
            ax.text(xi, v + eh + 0.012, f"{v:+.3f}", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold")
        else:
            ax.text(xi, v - 0.010, f"{v:+.3f}", ha="center", va="top",
                    fontsize=10.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta label-efficiency AUC (mAP@[.50:.95])")
    ax.set_title("Adaptive correction and filtering policy contrasts")
    ax.set_ylim(-0.08, 0.40)
    ax.margins(x=0.04)
    fig.text(0.5, -0.02,
             "Independent retraining/rescoring; correct-only duplicates the "
             "sigma=0 ref policy; contrasts are not same-frame causal shares.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    out = save_fig(plt, OUTDIR / "fig_mechanism_decomp.png")
    plt.close(fig)
    return out


def fig_verify_vs_correct(plt, style):
    from _figstyle import save_fig

    i1 = EXP / "clinician_i1_verify" / "clinician_curve.csv"
    i4 = EXP / "clinician_i4_uniform30" / "clinician_curve.csv"
    a_mu, a_sd = _auc_mean_std(i4, "A")
    ref_mu, ref_sd = _auc_mean_std(i4, "ref")
    selective_mu, selective_sd = _auc_mean_std(i1, "B", "0.0")
    fullb_mu, fullb_sd = _auc_mean_std(EXP / "clinician_i2_decomp" / "clinician_curve.csv",
                                       "ref_filter")

    labels = ["A\n(after shared GT init)", "Selective correction\n(GT-OKS oracle)",
              "Full B\n(correct + reject)", "ref\n(full GT)"]
    vals = [a_mu, selective_mu, fullb_mu, ref_mu]
    sds = [a_sd, selective_sd, fullb_sd, ref_sd]
    colors = [style["A"]["color"], "#E8A33D", style["B"]["color"], style["ref"]["color"]]
    recov = [None, "45% of\nA-to-ref gap", "97% of\nA-to-ref gap", "reference"]

    fig, ax = plt.subplots(figsize=(7.4, 5))
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=sds, color=colors, width=0.6, edgecolor="black",
           linewidth=0.6, capsize=5, zorder=3,
           error_kw=dict(elinewidth=1.3, ecolor="black"))
    for xi, v, sd, rc in zip(x, vals, sds, recov):
        ax.text(xi, v + sd + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
        if rc:
            ax.text(xi, v / 2, rc, ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.axhline(ref_mu, color=style["ref"]["color"], linestyle=":", linewidth=1.4,
               zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Label-efficiency AUC (mAP@[.50:.95])")
    ax.set_title("GT-OKS selective-correction oracle closes 45% of the A-to-ref gap")
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    out = save_fig(plt, OUTDIR / "fig_verify_vs_correct.png")
    plt.close(fig)
    return out


def fig_backbone_recovery(plt, style):
    from _figstyle import save_fig

    rows = [
        ("yolo11n", 97, 94, 100),
        ("yolo11s", 102, 100, 104),
        ("yolo11m", 99, 96, 101),
    ]
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    lo = [r[1] - r[2] for r in rows]
    hi = [r[3] - r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 5))
    x = np.arange(len(names))
    ax.errorbar(x, vals, yerr=[lo, hi], fmt="o", markersize=9,
                color=style["B"]["color"], markeredgecolor="black",
                markeredgewidth=0.6, capsize=6, linewidth=1.6, zorder=3)
    ax.axhline(100, color=style["ref"]["color"], linestyle="--", linewidth=1.6,
               zorder=2, label="Full-GT comparator (100% gap recovery)")
    for xi, v, h in zip(x, vals, hi):
        ax.text(xi, v + h + 1.0, f"{v}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Recovery of the A-to-ref gap (%)")
    ax.set_title("Recovery of the A-to-ref gap across backbones")
    ax.set_ylim(88, 108)
    ax.legend(loc="lower right", framealpha=1.0)
    fig.tight_layout()
    out = save_fig(plt, OUTDIR / "fig_backbone_recovery.png")
    plt.close(fig)
    return out


_DOWN_COND_LABEL = {
    "A": "A: pseudo-label after GT init",
    "B": "B: simulated correction + reject",
    "ref": "ref: full-GT comparator",
}
_DOWN_ORDER = ["A", "B", "ref"]


def fig_downstream_rom_smooth(plt, style):
    from _figstyle import save_fig

    data, budgets = _load_downstream(EXP / "downstream_11n_u30")
    conds = [c for c in _DOWN_ORDER if c in data["rom_error_mean"]]

    panels = [
        ("rom_error_mean", "ROM error (deg, lower=better)"),
        (
            "smoothness_error_mean",
            "Mean jerk-magnitude error (image diagonal s$^{-3}$)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for panel_idx, (ax, (metric, ylab)) in enumerate(zip(axes, panels)):
        for c in conds:
            ys, es = [], []
            for b in budgets:
                mu, sd = _ms(data[metric][c].get(b, []))
                ys.append(mu); es.append(sd if sd == sd else 0.0)
            ys = np.array(ys); es = np.array(es)
            st = dict(style[c])
            lw = 2.6 if c == "B" else 1.9
            ax.plot(budgets, ys, linewidth=lw, label=_DOWN_COND_LABEL[c], **st)
            ax.fill_between(budgets, ys - es, ys + es, alpha=0.15,
                            color=st.get("color"))
        ax.set_xlabel("Label budget (fraction of pool)")
        ax.set_ylabel(ylab)
        ax.text(0.02, 0.96, f"({'ab'[panel_idx]})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
    axes[0].set_title("Downstream ROM error vs budget (11n)")
    axes[1].set_title("Downstream smoothness error vs budget (11n)")
    axes[0].legend(loc="upper right", framealpha=1.0)
    fig.tight_layout()
    out = save_fig(plt, OUTDIR / "fig_downstream_rom_smooth.png")
    plt.close(fig)
    return out


def fig_downstream_angle(plt, style, ddir, tag, title):
    from _figstyle import save_fig

    data, budgets = _load_downstream(ddir)
    m = "joint_angle_mae_mean"
    conds = [c for c in _DOWN_ORDER if c in data[m]]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in conds:
        ys, es = [], []
        for b in budgets:
            mu, sd = _ms(data[m][c].get(b, []))
            ys.append(mu); es.append(sd if sd == sd else 0.0)
        ys = np.array(ys); es = np.array(es)
        st = dict(style[c])
        lw = 2.6 if c == "B" else 1.9
        ax.plot(budgets, ys, linewidth=lw, label=_DOWN_COND_LABEL[c], **st)
        ax.fill_between(budgets, ys - es, ys + es, alpha=0.15, color=st.get("color"))
    ax.set_xlabel("Label budget (fraction of pool)")
    ax.set_ylabel("Joint-angle MAE (deg)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=1.0)
    fig.tight_layout()
    out = save_fig(plt, OUTDIR / f"fig_downstream_{tag}.png")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    global OUTDIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(OUTDIR),
                    help="output directory for the figures")
    args = ap.parse_args(argv)

    OUTDIR = Path(args.outdir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _figstyle import apply_pub_style, COND_STYLE
    apply_pub_style(plt)
    style = COND_STYLE

    produced = []
    produced.append(fig_mechanism_decomp(plt, style))
    produced.append(fig_verify_vs_correct(plt, style))
    produced.append(fig_backbone_recovery(plt, style))
    produced.append(fig_downstream_rom_smooth(plt, style))
    produced.append(fig_downstream_angle(
        plt, style, EXP / "downstream_s_u30", "s",
        "Downstream joint-angle MAE vs budget (yolo11s)"))
    produced.append(fig_downstream_angle(
        plt, style, EXP / "downstream_m_u30", "m",
        "Downstream joint-angle MAE vs budget (yolo11m)"))

    print("Wrote figures:")
    for p in produced:
        print(f"  {p}")
        print(f"  {Path(p).with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
