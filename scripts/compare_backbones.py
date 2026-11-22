"""Compare simulated correction-plus-rejection results across backbones.

Reads each backbone's master ``clinician_curve.csv`` (same methodology:
6-seed, round0=30ep Option-A) and, when present, its downstream rehab-metric
JSON dir, then emits a single side-by-side comparison table
(``verification/backbone_comparison.md``).

Backbones are passed as ``label:clinician_dir[:downstream_dir]`` triples, e.g.::

    python scripts/compare_backbones.py \
        yolo11n:outputs/experiments/clinician_optA6:outputs/experiments/downstream_optA6 \
        yolo11s:outputs/experiments/clinician_s:outputs/experiments/downstream_s \
        yolo11m:outputs/experiments/clinician_m:outputs/experiments/downstream_m

Only the pose-mAP columns require the clinician dir; downstream columns are
filled per backbone only when that downstream dir exists with *_downstream.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.metrics.learning_curve import area_under_learning_curve  # noqa: E402
from rehabal_pose.metrics.significance import bootstrap_recovery_ci, compare_curves  # noqa: E402
from rehabal_pose.utils.logging import get_logger  # noqa: E402

_LOG = get_logger("backbone_compare")
_METRIC = "map50_95"


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _auc_by_seed(df: pd.DataFrame, cond: str) -> dict[int, float]:
    """seed -> normalized label-efficiency AUC for one condition (keyed to align)."""
    out: dict[int, float] = {}
    for seed, gs in df[df.condition == cond].groupby("seed"):
        gs = gs.sort_values("budget")
        bs = gs["budget"].astype(float).tolist()
        vs = gs[_METRIC].astype(float).tolist()
        if len(bs) >= 2:
            out[int(seed)] = float(area_under_learning_curve(bs, vs, normalize=True))
    return out


def _auc_mean_std(df: pd.DataFrame, cond: str) -> tuple[float, float]:
    arr = np.array(list(_auc_by_seed(df, cond).values()), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _downstream_angle_at(ddir: Path, budget: float) -> dict:
    """condition -> mean joint-angle MAE over seeds at *budget* (or {} if absent)."""
    files = sorted(ddir.glob("*_downstream.json")) if ddir and ddir.is_dir() else []
    if not files:
        return {}
    acc: dict = {"A": [], "B": [], "ref": []}
    for f in files:
        name = f.stem.replace("_downstream", "")
        cond, _, _seed = name.rpartition("_s")
        if cond not in acc:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for _bk, bv in d.get("budgets", {}).items():
            if "error" in bv:
                raise ValueError(f"invalid downstream result {f.name}:{_bk}: {bv['error']}")
            if abs(float(bv["budget"]) - budget) < 1e-9:
                v = bv.get("aggregate", {}).get("joint_angle_mae_mean", {})
                if isinstance(v, dict):
                    v = v.get("mean")
                if v is not None:
                    acc[cond].append(float(v))
    return {c: (float(np.mean(vs)) if vs else float("nan")) for c, vs in acc.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backbones", nargs="+",
                    help="label:clinician_dir[:downstream_dir] per backbone")
    ap.add_argument("--budget", type=float, default=0.4,
                    help="budget at which to report per-budget significance/MAE")
    ap.add_argument("--out-md", default="verification/backbone_comparison.md")
    args = ap.parse_args(argv)

    rows = []
    for spec in args.backbones:
        parts = spec.split(":")
        label, cdir = parts[0], parts[1]
        ddir = Path(parts[2]) if len(parts) > 2 else None
        csv = Path(cdir) / "clinician_curve.csv"
        if not csv.is_file():
            _LOG.warning("skip %s: no %s", label, csv)
            continue
        df = pd.read_csv(csv)
        # Restrict condition B to the sigma=0 operating point. A directory that
        # also holds a correction-noise sweep (e.g. clinician_i4_uniform30 with B
        # at sigma 0/5/10/20/40) would otherwise stack multiple sigmas per seed
        # and corrupt the AUC/significance. A/ref carry no sigma. Matches the
        # sigma-0 slice used in analyze_clinician_experiment.py.
        if "sigma" in df.columns:
            sig = pd.to_numeric(df["sigma"], errors="coerce").fillna(0.0)
            df = df[(df.condition != "B") | (sig == 0.0)].copy()
        n_seeds = df["seed"].nunique()
        a_by = _auc_by_seed(df, "A")
        b_by = _auc_by_seed(df, "B")
        r_by = _auc_by_seed(df, "ref")
        a_m, a_s = _auc_mean_std(df, "A")
        b_m, b_s = _auc_mean_std(df, "B")
        r_m, r_s = _auc_mean_std(df, "ref")
        ceil = r_m - a_m
        recov = (b_m - a_m) / ceil if ceil > 1e-9 else float("nan")
        # 95% seed-bootstrap CI (10,000 resamples) on the recovery ratio,
        # matching analyze_clinician_experiment.py (fixed RNG).
        common = sorted(set(a_by) & set(b_by) & set(r_by))
        recov_lo, recov_hi = float("nan"), float("nan")
        if len(common) >= 2:
            recov_lo, recov_hi = bootstrap_recovery_ci(
                np.array([a_by[k] for k in common]),
                np.array([b_by[k] for k in common]),
                np.array([r_by[k] for k in common]),
            )
        # significance B vs A / B vs ref at the chosen budget
        cdf = df.rename(columns={"condition": "strategy"})
        comp = compare_curves(cdf, metric=_METRIC, reference="B",
                              baselines=["A", "ref"],
                              group_cols=("strategy", "budget", "seed"))
        def _at(baseline):
            sub = comp[(comp.baseline == baseline)
                       & (np.isclose(comp.budget.astype(float), args.budget))]
            if not len(sub):
                return None
            r = sub.iloc[0]
            return dict(diff=float(r.mean_diff), lo=float(r.ci_lo),
                        hi=float(r.ci_hi), p=float(r.p_value))
        bva = _at("A"); bvr = _at("ref")
        ds = _downstream_angle_at(ddir, args.budget)
        rows.append(dict(label=label, n_seeds=n_seeds, a_m=a_m, a_s=a_s,
                         b_m=b_m, b_s=b_s, r_m=r_m, r_s=r_s, recov=recov,
                         recov_lo=recov_lo, recov_hi=recov_hi,
                         bva=bva, bvr=bvr, ds=ds))

    if not rows:
        _LOG.error("no backbones with a clinician_curve.csv found")
        return 1

    bud = f"{args.budget:g}"
    L = ["# Backbone comparison — Simulated Correction plus Rejection (6 seeds)",
         "",
         f"Same methodology across backbones (Option-A round0). Pose metric "
         f"**{_METRIC}**; per-budget stats reported at **budget {bud}**. AUC is "
         f"normalized label-efficiency AUC (mean ± std over seeds). Recovery = "
         f"fraction of the A→ref gap recovered by condition B.",
         ""]

    # AUC block
    L += ["## Label-efficiency AUC (normalized, mean ± std)", "",
          "| backbone | seeds | A (pseudo) | B (simulated correction + reject) | ref (full-GT comparator) | B recovery of A→ref gap (95% CI) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if r["recov_lo"] == r["recov_lo"]:  # not NaN
            recov_cell = (f"**{r['recov']:.0%}** [{r['recov_lo'] * 100:.0f}, "
                          f"{r['recov_hi'] * 100:.0f}]")
        else:
            recov_cell = f"**{r['recov']:.0%}**"
        L.append(f"| {r['label']} | {r['n_seeds']} | {r['a_m']:.4f} ± {r['a_s']:.4f} | "
                 f"**{r['b_m']:.4f} ± {r['b_s']:.4f}** | {r['r_m']:.4f} ± {r['r_s']:.4f} | "
                 f"{recov_cell} |")
    L.append("")

    # significance block at budget
    L += [f"## B vs A and B vs ref at budget {bud} (exploratory unadjusted {_METRIC} mean difference, 95% CI, two-sided p)",
          "",
          "| backbone | B−A diff | 95% CI | p | B−ref diff | 95% CI | p |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        a = r["bva"]; v = r["bvr"]
        a_str = (f"+{a['diff']:.3f} | [{a['lo']:.3f}, {a['hi']:.3f}] | {a['p']:.3f}"
                 if a else "— | — | —")
        v_str = (f"{v['diff']:+.3f} | [{v['lo']:.3f}, {v['hi']:.3f}] | {v['p']:.3f}"
                 if v else "— | — | —")
        L.append(f"| {r['label']} | {a_str} | {v_str} |")
    L.append("")

    # downstream block (only if any backbone has it)
    if any(r["ds"] for r in rows):
        L += [f"## Downstream joint-angle MAE at budget {bud} (deg, lower=better)",
              "",
              "| backbone | A | B | ref |",
              "|---|---|---|---|"]
        for r in rows:
            ds = r["ds"]
            if ds:
                L.append(f"| {r['label']} | {ds.get('A', float('nan')):.2f} | "
                         f"**{ds.get('B', float('nan')):.2f}** | "
                         f"{ds.get('ref', float('nan')):.2f} |")
            else:
                L.append(f"| {r['label']} | — | — | — |")
        L.append("")

    L += ["## Interpretation notes", "",
          "- Recovery CIs are 95% bootstrap intervals from 10,000 seed resamples. "
          "Cross-backbone interval overlap is descriptive and is not a formal "
          "test of a capacity-by-condition interaction. Intervals that include "
          "100% do not establish equivalence "
          "to the full-GT comparator, and values above 100% do not make ref a "
          "mathematical upper bound.",
          f"- Two-sided permutation p floors at 2/2^n_seeds; with 6 seeds that is "
          f"2/64 = 0.031. These per-budget and per-backbone tests are exploratory "
          f"and unadjusted for multiplicity; emphasize effect sizes and CIs.",
          "- All three backbones share the round-0 GT init per seed, so A/B/ref "
          "comparisons within a backbone start from the same initialization. "
          "Later selected sets differ through condition-specific feedback.",
          "- The original 5-seed yolo11n run (round0=10ep) is reported separately "
          "(verification/clinician_results.md) and is NOT mixed into this 30ep table."]

    md = "\n".join(L) + "\n"
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    _LOG.info("Wrote %s", out)
    _safe_print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
