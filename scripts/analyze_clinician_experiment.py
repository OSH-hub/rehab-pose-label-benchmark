"""Usage: python scripts/analyze_clinician_experiment.py"""

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
from rehabal_pose.metrics.significance import compare_curves, summarize_win_region  # noqa: E402
from rehabal_pose.utils.logging import get_logger  # noqa: E402

_LOG = get_logger("clinician_analysis")


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


COND_LABEL = {
    "A": "A: pseudo-label after shared GT init",
    "B": "B: AL + simulated correction + segmentation-based reject",
    "ref": "ref: AL + full-GT comparator",
}


def _per_seed_auc(df: pd.DataFrame, metric: str) -> dict:
    out: dict = {}
    for cond, g in df.groupby("condition"):
        out[cond] = {}
        for seed, gs in g.groupby("seed"):
            gs = gs.sort_values("budget")
            bs = gs["budget"].astype(float).tolist()
            vs = gs[metric].astype(float).tolist()
            if len(bs) >= 2:
                out[cond][int(seed)] = float(
                    area_under_learning_curve(bs, vs, normalize=True)
                )
    return out


def _mean_std(vals) -> tuple:
    arr = np.array([v for v in vals if v == v], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _state_process_rows(
    experiment_dir: Path,
    seeds: list[int],
    budgets: list[float],
    sigma: float,
) -> dict[float, list[dict]]:
    by_budget: dict[float, list[dict]] = {float(b): [] for b in budgets}
    sig_token = f"{float(sigma):g}"
    for seed in seeds:
        state_path = experiment_dir / f"B_s{int(seed)}_sig{sig_token}" / "state.json"
        if not state_path.is_file():
            break
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            history = state.get("history", [])
            if len(history) != len(budgets):
                break
            for budget, row in zip(budgets, history):
                by_budget[float(budget)].append(row)
        except (OSError, ValueError, TypeError):
            break
    else:
        return by_budget

    aggregate_path = experiment_dir / "proposal_accounting.json"
    try:
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate_sigma = float(payload["condition_b_sigma_px"])
        records = payload["records"]
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    if not np.isclose(aggregate_sigma, sigma) or not isinstance(records, list):
        return {}

    by_budget = {float(b): [] for b in budgets}
    expected_seeds = {int(seed) for seed in seeds}
    post_initial_budgets = [float(b) for b in budgets[1:]]
    for row in records:
        if not isinstance(row, dict):
            return {}
        try:
            seed = int(row["seed"])
            row_budget = float(row["budget"])
            float(row["n_proposed"])
            float(row["n_rejected"])
        except (KeyError, ValueError, TypeError):
            return {}
        if seed not in expected_seeds:
            continue
        matches = [b for b in post_initial_budgets if np.isclose(row_budget, b)]
        if len(matches) != 1:
            return {}
        by_budget[matches[0]].append(row)

    for budget in post_initial_budgets:
        rows = by_budget[budget]
        if len(rows) != len(expected_seeds):
            return {}
        if {int(row["seed"]) for row in rows} != expected_seeds:
            return {}
    return by_budget


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="outputs/experiments/clinician/clinician_curve.csv")
    ap.add_argument("--metric", default="map50_95", choices=["map50_95", "map50"])
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="condition-B sigma slice to analyze (default 0.0)")
    ap.add_argument("--out-md", default="verification/clinician_results.md")
    ap.add_argument("--out-png", default="outputs/experiments/clinician/clinician_curve.png")
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        _LOG.error("No CSV at %s - run scripts/run_clinician_experiment.py first.", csv_path)
        return 1
    df = pd.read_csv(csv_path)

    is_b = df["condition"] == "B"
    if "sigma" in df.columns:
        keep = (~is_b) | (is_b & (pd.to_numeric(df["sigma"], errors="coerce")
                                  .fillna(0.0) == args.sigma))
        df = df[keep].copy()

    conds = [c for c in ["A", "B", "ref"] if c in set(df["condition"])]
    budgets = sorted(df["budget"].dropna().unique().tolist())
    seeds = sorted(df["seed"].unique().tolist())
    _LOG.info("Conditions=%s budgets=%s seeds=%s metric=%s sigma=%g",
              conds, budgets, seeds, args.metric, args.sigma)

    auc = _per_seed_auc(df, args.metric)
    lines = ["# Simulated Correction-plus-Rejection Experiment — Results",
             "",
             f"Metric: pose **{args.metric}**; seeds: {seeds}; condition-B sigma: {args.sigma}px.",
             "",
             "## Per-condition label-efficiency AUC (normalized, mean ± std over seeds)",
             "",
             "| condition | AUC | per-seed |",
             "|---|---|---|"]
    auc_summary = {}
    for c in conds:
        m, s = _mean_std(list(auc.get(c, {}).values()))
        auc_summary[c] = (m, s)
        per = ", ".join(f"{auc[c][sd]:.4f}" for sd in sorted(auc.get(c, {})))
        lines.append(f"| {COND_LABEL.get(c, c)} | {m:.4f} ± {s:.4f} | {per} |")
    lines.append("")

    lines += ["## Per-budget mean pose " + args.metric + " (over seeds)", "",
              "| budget | " + " | ".join(conds) + " |",
              "|---|" + "|".join(["---"] * len(conds)) + "|"]
    curve_means: dict = {c: [] for c in conds}
    for b in budgets:
        row = [f"{b:g}"]
        for c in conds:
            vals = df[(df.condition == c) & (np.isclose(df.budget.astype(float), b))][args.metric]
            mu = float(vals.astype(float).mean()) if len(vals) else float("nan")
            curve_means[c].append(mu)
            row.append(f"{mu:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    cdf = df.rename(columns={"condition": "strategy"}).copy()
    metric_col = args.metric
    lines += ["## Exploratory unadjusted comparisons (reference = B)", ""]
    try:
        baselines = [c for c in ["A", "ref"] if c in conds]
        comp = compare_curves(cdf, metric=metric_col, reference="B",
                              baselines=baselines,
                              group_cols=("strategy", "budget", "seed"))
        lines.append("```")
        lines.append(comp.to_string(index=False))
        lines.append("```")
        lines.append("")
        n_seeds_sig = len(seeds)
        n_perm = 2 ** n_seeds_sig if n_seeds_sig > 0 else 0
        p_floor = 2.0 / n_perm if n_perm else float("nan")
        lines.append(
            f"> **Statistical-floor note.** The two-sided sign-flip permutation test "
            f"enumerates all 2^{n_seeds_sig} = {n_perm} sign assignments, so its p-value "
            f"cannot fall below **2/{n_perm} = {p_floor:.4f}** with {n_seeds_sig} seeds. "
            f"This finite resolution limits how small the p-value can be. Interpret the "
            f"effect size and confidence interval as the primary evidence. These "
            f"comparisons are exploratory and unadjusted for multiplicity."
        )
        lines.append("")
        _raw_summary = summarize_win_region(comp).replace("RehabAL", "Condition B")
        lines.append("_Automated summary (interpret alongside the statistical-resolution "
                     "note above):_ " + " ".join(_raw_summary.split()))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"_compare_curves failed: {exc}_")
    lines.append("")

    if "B" in conds:
        bdf = df[df.condition == "B"]
        state_rows = _state_process_rows(csv_path.parent, seeds, budgets, args.sigma)
        lines += ["## Condition-B proposal and commit accounting (mean over seeds)", "",
                  "The loop overproposes and truncates valid excess candidates after the "
                  "target increment is reached, so committed plus rejected need not equal "
                  "proposed. Rejection rates are per-seed rejected/proposed rates averaged "
                  "over seeds; these are simulator process counts, not human workload.", "",
                  "| budget | proposed | committed accepted/corrected | rejected | rejected/proposed |",
                  "|---|---|---|---|---|"]
        for b in budgets:
            sub = bdf[np.isclose(bdf.budget.astype(float), b)]
            if not len(sub):
                continue
            na = sub.n_accepted.astype(float).mean()
            nr = sub.n_rejected.astype(float).mean()
            rows = state_rows.get(float(b), [])
            np_mean = float(np.mean([float(r.get("n_proposed", 0)) for r in rows])) if rows else float("nan")
            rr_values = [
                float(r.get("n_rejected", 0)) / float(r["n_proposed"])
                for r in rows if float(r.get("n_proposed", 0)) > 0
            ]
            rr = float(np.mean(rr_values)) if rr_values else float("nan")
            if np.isclose(b, budgets[0]):
                lines.append(
                    f"| {b:g} | 0 | {na:.0f} (shared GT initialization) | 0 | N/A |"
                )
            elif rows:
                lines.append(
                    f"| {b:g} | {np_mean:.1f} | {na:.1f} | {nr:.1f} | {rr:.1%} |"
                )
            else:
                lines.append(
                    f"| {b:g} | unavailable in master CSV | {na:.1f} | {nr:.1f} | N/A |"
                )
        lines.append("")

    b_collapsed = False
    if "B" in conds:
        bdf_all = df[df.condition == "B"]
        if "n_rejected" in bdf_all.columns:
            total_rejected = bdf_all["n_rejected"].astype(float).sum()
            if total_rejected == 0:
                b_collapsed = True
                warn_msg = (
                    "\n> [!WARNING] **CONDITION-B VALIDATION FAILED**: condition B's total "
                    "n_rejected == 0 across all rounds and seeds. The SimulatedClinician "
                    "did not reject any frame; B has collapsed into ref. The rejection "
                    "mechanism was inactive because the segmentation table was empty or "
                    "occlusion flags were absent. "
                    "**These B results are invalid for the simulated correction-plus-rejection "
                    "condition.** Check the Segmentation.csv path and rerun.\n"
                )
                lines.append(warn_msg)
                _LOG.warning(
                    "Condition-B validation failed: n_rejected==0 across all rounds/seeds. "
                    "B collapsed into ref because the rejection effect is inactive."
                )

    if {"A", "B", "ref"} <= set(conds):
        a_m = auc_summary["A"][0]; b_m = auc_summary["B"][0]; r_m = auc_summary["ref"][0]
        gap = r_m - a_m
        recovered = (b_m - a_m) / gap if gap > 1e-9 else float("nan")
        n_seeds = len(seeds)
        common = [sd for sd in seeds
                  if sd in auc.get("A", {}) and sd in auc.get("B", {})
                  and sd in auc.get("ref", {})]
        d_lo = d_hi = rec_lo = rec_hi = float("nan")
        if len(common) >= 2:
            aA = np.array([auc["A"][sd] for sd in common], float)
            aB = np.array([auc["B"][sd] for sd in common], float)
            aR = np.array([auc["ref"][sd] for sd in common], float)
            rng = np.random.default_rng(0)
            bidx = rng.integers(0, len(common), size=(10000, len(common)))
            d_boot = aB[bidx].mean(1) - aA[bidx].mean(1)
            g_boot = aR[bidx].mean(1) - aA[bidx].mean(1)
            rec_boot = np.divide(d_boot, g_boot, out=np.full_like(d_boot, np.nan),
                                 where=np.abs(g_boot) > 1e-9)
            d_lo, d_hi = np.percentile(d_boot, [2.5, 97.5])
            rec_boot = rec_boot[np.isfinite(rec_boot)]
            if rec_boot.size:
                rec_lo, rec_hi = np.percentile(rec_boot, [2.5, 97.5])
        validity_note = (
            "**[INVALID: B==ref because the rejection mechanism is inactive]** "
            if b_collapsed else ""
        )
        lines += ["## Primary aggregate comparison", "",
                  f"- AUC: A={a_m:.4f}, B={b_m:.4f} {validity_note}, ref={r_m:.4f}.",
                  f"- Condition B over pseudo-label condition A: normalized-AUC "
                  f"delta={b_m - a_m:+.4f}, 95% seed-bootstrap CI "
                  f"[{d_lo:+.4f}, {d_hi:+.4f}] (10,000 resamples over seeds; CI excludes 0 "
                  f"indicates a positive difference across the sampled seeds).",
                  f"- A-to-ref gap recovered by condition B: "
                  f"{recovered:.0%} (A-to-ref gap={gap:.4f}), "
                  f"95% seed-bootstrap CI [{rec_lo:.0%}, {rec_hi:.0%}]. An interval that "
                  f"includes 100% does not establish equality or equivalence with the GT "
                  f"comparator; ref is not treated as a mathematical upper bound.",
                  "",
                  f"_With {n_seeds} seed(s), effect size and the bootstrap CI should be "
                  f"emphasized because the two-sided permutation p-value is bounded below "
                  f"by 2/2^{n_seeds}._"]

    md = "\n".join(lines) + "\n"
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    _LOG.info("Wrote %s", out_md)
    _safe_print(md)

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _figstyle import apply_pub_style, style_for, metric_label, save_fig
        apply_pub_style(plt)
        plt.figure(figsize=(7, 5))
        for c in conds:
            ys, es = [], []
            for b in budgets:
                vals = df[(df.condition == c) & (np.isclose(df.budget.astype(float), b))][args.metric].astype(float)
                ys.append(vals.mean() if len(vals) else np.nan)
                es.append(vals.std(ddof=0) if len(vals) > 1 else 0.0)
            ys = np.array(ys); es = np.array(es)
            st = style_for(c)
            plt.plot(budgets, ys, linewidth=2.5 if c == "B" else 1.8,
                     label=COND_LABEL.get(c, c), **st)
            plt.fill_between(budgets, ys - es, ys + es, alpha=0.15, color=st.get("color"))
        plt.xlabel("Label budget (fraction of pool)")
        plt.ylabel(f"Pose {metric_label(args.metric)}")
        plt.title("Correction-in-the-Loop: label efficiency (REHAB24-6)")
        plt.legend(framealpha=1.0)
        plt.tight_layout()
        out_png = save_fig(plt, args.out_png)
        _LOG.info("Wrote %s", out_png)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("plot skipped: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
