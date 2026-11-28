"""Usage: python scripts/analyze_sigma_ablation.py --csv outputs/experiments/clinician_optA6/clinician_curve.csv"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.metrics.learning_curve import area_under_learning_curve  # noqa: E402
from rehabal_pose.metrics.significance import bootstrap_recovery_ci, compare_curves  # noqa: E402
from rehabal_pose.utils.logging import get_logger  # noqa: E402

_LOG = get_logger("sigma_ablation")
_METRIC = "map50_95"


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _auc_per_seed(df: pd.DataFrame) -> list[float]:
    return list(_auc_by_seed(df).values())


def _auc_by_seed(df: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed, gs in df.groupby("seed"):
        gs = gs.sort_values("budget")
        bs = gs["budget"].astype(float).tolist()
        vs = gs[_METRIC].astype(float).tolist()
        if len(bs) >= 2:
            out[int(seed)] = float(area_under_learning_curve(bs, vs, normalize=True))
    return out


def _ms(vals) -> tuple[float, float]:
    a = np.array([v for v in vals if v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0))) if a.size else (float("nan"), float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/experiments/clinician_optA6/clinician_curve.csv")
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--out-md", default="verification/sigma_ablation.md")
    ap.add_argument("--out-png", default="outputs/experiments/clinician_optA6/sigma_ablation.png")
    args = ap.parse_args(argv)

    df = pd.read_csv(Path(args.csv))
    sig = pd.to_numeric(df.get("sigma"), errors="coerce")
    is_b = df.condition == "B"

    a_by = _auc_by_seed(df[df.condition == "A"])
    r_by = _auc_by_seed(df[df.condition == "ref"])
    a_auc = _ms(list(a_by.values()))
    r_auc = _ms(list(r_by.values()))

    sigmas = sorted(pd.unique(sig[is_b].dropna()))
    if not sigmas:
        _LOG.error("no condition-B sigma rows in %s", args.csv)
        return 1

    rows = []
    for s in sigmas:
        bdf = df[is_b & (sig == s)]
        b_by = _auc_by_seed(bdf)
        b_m, b_sd = _ms(list(b_by.values()))
        ceil = r_auc[0] - a_auc[0]
        recov = (b_m - a_auc[0]) / ceil if ceil > 1e-9 else float("nan")
        common = sorted(set(a_by) & set(b_by) & set(r_by))
        recov_lo, recov_hi = float("nan"), float("nan")
        if len(common) >= 2:
            recov_lo, recov_hi = bootstrap_recovery_ci(
                np.array([a_by[k] for k in common]),
                np.array([b_by[k] for k in common]),
                np.array([r_by[k] for k in common]),
            )
        pair = pd.concat([bdf, df[df.condition == "A"], df[df.condition == "ref"]])
        cdf = pair.rename(columns={"condition": "strategy"})
        p_a = p_r = d_a = d_r = float("nan")
        try:
            comp = compare_curves(cdf, metric=_METRIC, reference="B",
                                  baselines=["A", "ref"],
                                  group_cols=("strategy", "budget", "seed"))
            for _, r in comp[np.isclose(comp.budget.astype(float), args.budget)].iterrows():
                if r.baseline == "A":
                    d_a, p_a = float(r.mean_diff), float(r.p_value)
                elif r.baseline == "ref":
                    d_r, p_r = float(r.mean_diff), float(r.p_value)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("sigma %g significance failed: %s", s, exc)
        rows.append(dict(sigma=s, b_m=b_m, b_sd=b_sd, recov=recov,
                         recov_lo=recov_lo, recov_hi=recov_hi,
                         d_a=d_a, p_a=p_a, d_r=d_r, p_r=p_r))

    bud = f"{args.budget:g}"
    L = ["# Simulated correction-quality ablation (sigma sweep)",
         "",
         f"Backbone-fixed condition-B label efficiency vs independent per-coordinate "
         f"Gaussian noise standard deviation **sigma (px)**. A (pseudo-label) and ref "
         f"(full-GT comparator) are sigma-independent "
         f"anchors. Pose metric **{_METRIC}**; exploratory unadjusted comparisons at budget "
         f"**{bud}**. AUC is normalized label-efficiency AUC (mean ± std over seeds).",
         "",
         f"- **A anchor (pseudo-label):** AUC = {a_auc[0]:.4f} ± {a_auc[1]:.4f}",
         f"- **ref anchor (full GT):**   AUC = {r_auc[0]:.4f} ± {r_auc[1]:.4f}",
         "",
          "| per-coordinate sigma (px) | B AUC | A-to-ref gap recovery (95% CI) | B−A diff at budget "
         f"{bud} | p | B−ref diff at budget {bud} | p |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["recov_lo"] == r["recov_lo"]:
            recov_cell = (f"{r['recov']:.0%} [{r['recov_lo'] * 100:.0f}, "
                          f"{r['recov_hi'] * 100:.0f}]")
        else:
            recov_cell = f"{r['recov']:.0%}"
        L.append(
            f"| {r['sigma']:g} | {r['b_m']:.4f} ± {r['b_sd']:.4f} | {recov_cell} | "
            f"+{r['d_a']:.3f} | {r['p_a']:.3f} | {r['d_r']:+.3f} | {r['p_r']:.3f} |")
    L += ["",
          "## Reading",
          "- B−A effect sizes across sigma test whether the observed gain is "
          "confined to exact ground-truth corrections. The per-sigma p-values "
          "are exploratory and unadjusted for multiplicity.",
          "- A-to-ref gap recovery traces sensitivity to simulated correction "
          "quality; ref is a comparator, not a mathematical upper bound.",
          f"- Recovery CIs are 95% seed-bootstrap intervals (10,000 resamples, fixed "
          f"RNG). These intervals describe uncertainty in the recovery ratio and "
          f"are not equivalence or non-inferiority tests against full GT.",
          "- Condition B uses a simulated GT-based oracle with image-bounded "
          "independent x/y Gaussian noise; sigma is the per-coordinate standard "
          "deviation, not Euclidean keypoint error. This is not a human study."]
    md = "\n".join(L) + "\n"
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    _LOG.info("Wrote %s", out)
    _safe_print(md)

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _figstyle import apply_pub_style, style_for, metric_label, save_fig
        apply_pub_style(plt)
        xs = [r["sigma"] for r in rows]
        ys = np.array([r["b_m"] for r in rows])
        es = np.array([r["b_sd"] for r in rows])
        stB, stA, stR = style_for("B"), style_for("A"), style_for("ref")
        plt.figure(figsize=(7, 5))
        plt.plot(
            xs,
            ys,
            linewidth=2.5,
            label="B: simulated correction + segmentation-based reject",
            **stB,
        )
        plt.fill_between(xs, ys - es, ys + es, alpha=0.15, color=stB["color"])
        plt.axhline(a_auc[0], color=stA["color"], linestyle=stA["linestyle"],
                    label="A: pseudo-label after shared GT init")
        plt.axhline(r_auc[0], color=stR["color"], linestyle=stR["linestyle"],
                    label="ref: full-GT comparator")
        plt.xlabel("Per-coordinate Gaussian noise SD σ (px)")
        plt.ylabel(f"Label-efficiency AUC ({metric_label(_METRIC)})")
        plt.title("Label efficiency vs simulated correction quality (REHAB24-6)")
        plt.legend(framealpha=1.0)
        plt.tight_layout()
        op = save_fig(plt, args.out_png)
        _LOG.info("Wrote %s", op)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("plot skipped: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
