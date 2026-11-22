"""Analyze the three supplemental ablation experiments (I2/I1/I3) against the u30 anchor.

This script reads the result CSVs for three supplemental ablations together
with the uniform-30-epoch anchor (u30) and produces a confidence-interval-backed
Markdown summary:

1. **I2 filter x correct 2x2 decomposition** — split condition B's total AUC
   gain (B - A) into a *filtering* contribution (reject bad pseudo-labels only)
   and a *correction* contribution (GT-correct accepted frames only), plus the
   interaction. Answers "which half of the mechanism carries the effect?".
2. **I1 selective correction vs always-correct B** — the always-correct arm
   applies GT-based correction to every accepted frame; the thresholded arm
   keeps predictions with OKS >= 0.5 and corrects predictions below that level.
3. **I3 A vs A_conf** — compare the confidence-gated self-training baseline
   (A_conf) with A, and compare simulated condition B with A_conf.

Each analysis reports normalized label-efficiency AUC (mean +/- std over seeds)
and 95% seed-bootstrap CIs, reusing
``rehabal_pose.metrics.learning_curve.area_under_learning_curve`` and
``rehabal_pose.metrics.significance.{paired_bootstrap_ci,bootstrap_recovery_ci}``.

Missing result files are reported and skipped so the script can summarize a
partially available set of ablations.

Usage::
    python scripts/analyze_review_experiments.py
    python scripts/analyze_review_experiments.py --metric map50_95 --out-md verification/review_experiments_results.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from rehabal_pose.metrics.learning_curve import area_under_learning_curve  # noqa: E402
from rehabal_pose.metrics.significance import (  # noqa: E402
    bootstrap_recovery_ci,
    paired_bootstrap_ci,
)

# ---------------------------------------------------------------------------
# Default result locations (completed experiments will populate these).
# ---------------------------------------------------------------------------
DEFAULT_I2 = "outputs/experiments/clinician_i2_decomp/clinician_curve.csv"
DEFAULT_I1 = "outputs/experiments/clinician_i1_verify/clinician_curve.csv"
DEFAULT_I3 = "outputs/experiments/clinician_i3_conf/clinician_curve.csv"
DEFAULT_ANCHOR = "outputs/experiments/clinician_i4_uniform30/clinician_curve.csv"
DEFAULT_OUT_MD = "verification/review_experiments_results.md"

# Condition-B rows carry a jitter sigma; the anchor's canonical B slice is sigma=0.
ANCHOR_B_SIGMA = 0.0


def _safe_print(text: str) -> None:
    """Write *text* to stdout, falling back to utf-8 with replacement on cp949 consoles."""
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def load_curve(path: str | Path) -> Optional[pd.DataFrame]:
    """Load a ``clinician_curve.csv`` if present, else ``None`` (experiment unfinished)."""
    p = Path(path)
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    return df


def _fmt(x: float, nd: int = 4) -> str:
    return "nan" if x != x else f"{x:.{nd}f}"


def _fmt_pct(x: float) -> str:
    return "nan" if x != x else f"{x:.0%}"


# ---------------------------------------------------------------------------
# Core: per-condition per-seed normalized-AUC vector aligned to a seed list.
# ---------------------------------------------------------------------------


def auc_by_seed(
    df: pd.DataFrame,
    condition: str,
    metric: str,
    *,
    sigma: Optional[float] = None,
) -> dict[int, float]:
    """Map ``seed -> normalized label-efficiency AUC`` for one condition.

    ``sigma`` (when given) restricts to a single condition-B jitter slice; the
    ``sigma`` column is coerced to numeric with empty -> 0 (A/ref rows carry no
    sigma). Seeds with < 2 budget points are dropped (AUC undefined).
    """
    sub = df[df["condition"] == condition]
    if sigma is not None and "sigma" in sub.columns:
        sig = pd.to_numeric(sub["sigma"], errors="coerce").fillna(0.0)
        sub = sub[np.isclose(sig, sigma)]
    out: dict[int, float] = {}
    for seed, g in sub.groupby("seed"):
        g = g.sort_values("budget")
        bs = g["budget"].astype(float).to_numpy()
        vs = g[metric].astype(float).to_numpy()
        if bs.size >= 2:
            out[int(seed)] = float(area_under_learning_curve(bs, vs, normalize=True))
    return out


def aligned_vectors(
    seed_maps: dict[str, dict[int, float]],
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Given ``{name: {seed: auc}}`` return the common seeds (present & finite in
    ALL names) and one aligned float vector per name over those seeds.

    Using a single common-seed set keeps every mean/delta computed on the same
    paired seeds, so additive decompositions close exactly.
    """
    names = list(seed_maps)
    if not names:
        return [], {}
    common = set(seed_maps[names[0]])
    for n in names[1:]:
        common &= set(seed_maps[n])
    seeds = sorted(common)
    vecs = {
        n: np.array([seed_maps[n][s] for s in seeds], dtype=np.float64) for n in names
    }
    return seeds, vecs


def _mean_std(v: np.ndarray) -> tuple[float, float]:
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=0))


# ---------------------------------------------------------------------------
# Analysis 1 — I2 filter x correct 2x2 decomposition
# ---------------------------------------------------------------------------


def analyze_i2_decomp(
    decomp_df: Optional[pd.DataFrame],
    anchor_df: Optional[pd.DataFrame],
    *,
    metric: str = "map50_95",
) -> dict:
    """2x2 decomposition of condition B into filtering + correction parts.

    Cells (both AUC, normalized):
      * ``A``               (no filter, no correct)  <- anchor u30, condition ``A``
      * ``A_filter``        (filter only)            <- decomp, condition ``A_filter``
      * ``B_correct_only``  (correct only)           <- decomp, condition ``B_correct_only``
      * ``B`` = ``ref_filter`` (filter + correct)    <- decomp, condition ``ref_filter``

    Total gain ``B - A`` = filtering ``(A_filter - A)`` + correction
    ``(B_correct_only - A)`` + interaction. All four cell means are taken over the
    SAME common seed set, so the identity closes exactly.
    """
    if decomp_df is None or anchor_df is None:
        return {"skipped": True, "reason": "i2_decomp and/or anchor CSV not found"}

    maps = {
        "A": auc_by_seed(anchor_df, "A", metric),
        "A_filter": auc_by_seed(decomp_df, "A_filter", metric),
        "B_correct_only": auc_by_seed(decomp_df, "B_correct_only", metric),
        "B": auc_by_seed(decomp_df, "ref_filter", metric),
    }
    if any(len(m) == 0 for m in maps.values()):
        missing = [k for k, m in maps.items() if len(m) == 0]
        return {"skipped": True, "reason": f"no data for condition(s): {missing}"}

    seeds, v = aligned_vectors(maps)
    if len(seeds) < 2:
        return {"skipped": True, "reason": f"only {len(seeds)} common seed(s)"}

    mA = float(v["A"].mean())
    mAf = float(v["A_filter"].mean())
    mBc = float(v["B_correct_only"].mean())
    mB = float(v["B"].mean())

    total = mB - mA
    filtering = mAf - mA
    correction = mBc - mA
    interaction = total - filtering - correction

    # Paired seed-bootstrap CIs for the total gain and each contribution.
    ci_total = paired_bootstrap_ci(v["B"], v["A"], seed=0)
    ci_filt = paired_bootstrap_ci(v["A_filter"], v["A"], seed=0)
    ci_corr = paired_bootstrap_ci(v["B_correct_only"], v["A"], seed=0)

    # Optional supplementary condition: reject-only (present in some sweeps).
    reject_map = auc_by_seed(decomp_df, "B_reject_only", metric)
    reject_mean = _mean_std(np.array(list(reject_map.values())))[0] if reject_map else float("nan")

    return {
        "skipped": False,
        "seeds": seeds,
        "means": {"A": mA, "A_filter": mAf, "B_correct_only": mBc, "B": mB},
        "stds": {k: _mean_std(v[k])[1] for k in v},
        "total_gain": total,
        "filtering": filtering,
        "correction": correction,
        "interaction": interaction,
        "ci_total": (ci_total[0], ci_total[1]),
        "ci_filtering": (ci_filt[0], ci_filt[1]),
        "ci_correction": (ci_corr[0], ci_corr[1]),
        "reject_only_mean": reject_mean,
        "vectors": v,
    }


def _md_i2(res: dict) -> list[str]:
    lines = ["## I2 - Filtering x Correction 2x2 decomposition", ""]
    if res.get("skipped"):
        lines += [f"_Skipped: {res['reason']}._", ""]
        return lines
    m = res["means"]
    s = res["stds"]
    lines += [
        f"Common seeds: {res['seeds']} (all four cells on the same paired seeds).",
        "",
        "| cell | mechanism | AUC (mean +/- std) |",
        "|---|---|---|",
        f"| A | no filter, no correct | {_fmt(m['A'])} +/- {_fmt(s['A'])} |",
        f"| A_filter | filter only | {_fmt(m['A_filter'])} +/- {_fmt(s['A_filter'])} |",
        f"| B_correct_only | correct only | {_fmt(m['B_correct_only'])} +/- {_fmt(s['B_correct_only'])} |",
        f"| B (=ref_filter) | filter + correct | {_fmt(m['B'])} +/- {_fmt(s['B'])} |",
    ]
    if res["reject_only_mean"] == res["reject_only_mean"]:
        lines.append(
            f"| B_reject_only (suppl.) | reject only | {_fmt(res['reject_only_mean'])} |"
        )
    lines += [
        "",
        "**Decomposition of the total condition-B gain (B - A):**",
        "",
        "| component | delta AUC | 95% seed-bootstrap CI |",
        "|---|---|---|",
        f"| total gain (B - A) | {_fmt(res['total_gain'])} | [{_fmt(res['ci_total'][0])}, {_fmt(res['ci_total'][1])}] |",
        f"| filtering (A_filter - A) | {_fmt(res['filtering'])} | [{_fmt(res['ci_filtering'][0])}, {_fmt(res['ci_filtering'][1])}] |",
        f"| correction (B_correct_only - A) | {_fmt(res['correction'])} | [{_fmt(res['ci_correction'][0])}, {_fmt(res['ci_correction'][1])}] |",
        f"| interaction (total - filter - correct) | {_fmt(res['interaction'])} | - |",
        "",
    ]
    # One-line interpretation.
    tot = res["total_gain"]
    filt = res["filtering"]
    corr = res["correction"]
    inter = res["interaction"]
    if tot > 1e-9:
        fp, cp = filt / tot, corr / tot
        dom = "filtering" if filt > corr else "correction"
        synergy = (
            "super-additive (synergistic)" if inter > 1e-4
            else "sub-additive" if inter < -1e-4
            else "additive"
        )
        lines.append(
            f"_Interpretation: {dom} carries the larger share "
            f"(filtering {fp:+.0%}, correction {cp:+.0%} of the total gain); the two "
            f"mechanisms combine {synergy} (interaction {inter:+.4f})._"
        )
    else:
        lines.append("_Interpretation: the total condition-B gain is ~0; decomposition is not informative._")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Analysis 2 — I1 OKS-gated selective correction vs always-correct B
# ---------------------------------------------------------------------------


def analyze_i1_verify(
    verify_df: Optional[pd.DataFrame],
    anchor_df: Optional[pd.DataFrame],
    *,
    metric: str = "map50_95",
) -> dict:
    """Compare always-correct B with OKS-gated selective-correction B.

    A and ref come from the anchor. The A-to-ref gap is the denominator used
    for descriptive recovery. Always-correct B is the
    anchor's sigma=0 B slice; selective-correction B is the ``B`` condition of
    the I1 run. Recovery = ``(B - A) / (ref - A)`` with a seed-bootstrap CI.
    """
    if verify_df is None or anchor_df is None:
        return {"skipped": True, "reason": "i1_verify and/or anchor CSV not found"}

    maps = {
        "A": auc_by_seed(anchor_df, "A", metric),
        "ref": auc_by_seed(anchor_df, "ref", metric),
        "B_legacy": auc_by_seed(anchor_df, "B", metric, sigma=ANCHOR_B_SIGMA),
        "B_verify": auc_by_seed(verify_df, "B", metric),
    }
    if any(len(m) == 0 for m in maps.values()):
        missing = [k for k, m in maps.items() if len(m) == 0]
        return {"skipped": True, "reason": f"no data for arm(s): {missing}"}

    seeds, v = aligned_vectors(maps)
    if len(seeds) < 2:
        return {"skipped": True, "reason": f"only {len(seeds)} common seed(s)"}

    mA, mRef = float(v["A"].mean()), float(v["ref"].mean())
    mLeg, mVer = float(v["B_legacy"].mean()), float(v["B_verify"].mean())
    gap = mRef - mA

    def _recovery(mb: float) -> float:
        return (mb - mA) / gap if abs(gap) > 1e-9 else float("nan")

    ci_leg = paired_bootstrap_ci(v["B_legacy"], v["A"], seed=0)
    ci_ver = paired_bootstrap_ci(v["B_verify"], v["A"], seed=0)
    rec_leg_ci = bootstrap_recovery_ci(v["A"], v["B_legacy"], v["ref"], seed=0)
    rec_ver_ci = bootstrap_recovery_ci(v["A"], v["B_verify"], v["ref"], seed=0)

    verify_beats_a = ci_ver[0] > 0
    return {
        "skipped": False,
        "seeds": seeds,
        "means": {"A": mA, "ref": mRef, "B_legacy": mLeg, "B_verify": mVer},
        "gap": gap,
        "recovery_legacy": _recovery(mLeg),
        "recovery_verify": _recovery(mVer),
        "delta_legacy": mLeg - mA,
        "delta_verify": mVer - mA,
        "ci_delta_legacy": (ci_leg[0], ci_leg[1]),
        "ci_delta_verify": (ci_ver[0], ci_ver[1]),
        "ci_recovery_legacy": rec_leg_ci,
        "ci_recovery_verify": rec_ver_ci,
        "verify_beats_a": bool(verify_beats_a),
    }


def _md_i1(res: dict) -> list[str]:
    lines = ["## I1 - OKS-gated selective correction vs always-correct B", ""]
    if res.get("skipped"):
        lines += [f"_Skipped: {res['reason']}._", ""]
        return lines
    m = res["means"]
    lines += [
        f"Common seeds: {res['seeds']}. A-to-ref gap = {_fmt(res['gap'])}.",
        "",
        "| arm | AUC | delta vs A | 95% CI (delta) | recovery of A-to-ref gap | 95% CI (recovery) |",
        "|---|---|---|---|---|---|",
        f"| A (pseudo-label) | {_fmt(m['A'])} | - | - | - | - |",
        f"| ref (full GT) | {_fmt(m['ref'])} | - | - | 100% | - |",
        f"| B always-correct (GT based) | {_fmt(m['B_legacy'])} | {_fmt(res['delta_legacy'])} | "
        f"[{_fmt(res['ci_delta_legacy'][0])}, {_fmt(res['ci_delta_legacy'][1])}] | "
        f"{_fmt_pct(res['recovery_legacy'])} | "
        f"[{_fmt_pct(res['ci_recovery_legacy'][0])}, {_fmt_pct(res['ci_recovery_legacy'][1])}] |",
        f"| B selective-correction (keep if OKS>=0.5; correct otherwise) | {_fmt(m['B_verify'])} | {_fmt(res['delta_verify'])} | "
        f"[{_fmt(res['ci_delta_verify'][0])}, {_fmt(res['ci_delta_verify'][1])}] | "
        f"{_fmt_pct(res['recovery_verify'])} | "
        f"[{_fmt_pct(res['ci_recovery_verify'][0])}, {_fmt_pct(res['ci_recovery_verify'][1])}] |",
        "",
    ]
    if res["verify_beats_a"] and res["recovery_verify"] >= 0.5:
        verdict = (
            "the selective-correction-minus-A delta CI excludes 0, and the arm recovers a "
            "majority of the A-to-ref gap. This arm still applies GT-based correction "
            "below the OKS threshold and is not a passive-verification control."
        )
    elif res["verify_beats_a"]:
        verdict = (
            "the selective-correction-minus-A delta CI excludes 0, but the arm recovers <50% "
            "of the A-to-ref gap. This estimate combines accepted predictions with GT-based "
            "corrections below the OKS threshold."
        )
    else:
        verdict = (
            "the selective-correction arm does not robustly beat A (delta CI includes 0)."
        )
    lines += [f"_Interpretation: {verdict}_", ""]
    return lines


# ---------------------------------------------------------------------------
# Analysis 3 — I3 A vs A_conf (and does B still win?)
# ---------------------------------------------------------------------------


def analyze_i3_conf(
    conf_df: Optional[pd.DataFrame],
    anchor_df: Optional[pd.DataFrame],
    *,
    metric: str = "map50_95",
) -> dict:
    """Compare confidence-gated self-training (A_conf) with A and B.

    A_conf comes from the i3 run; A, B (sigma=0), ref from the anchor. Two deltas
    with seed-bootstrap CIs: ``A_conf - A`` (does the stronger baseline help?) and
    ``B - A_conf`` (whether condition B remains above the stronger baseline).
    """
    if conf_df is None or anchor_df is None:
        return {"skipped": True, "reason": "i3_conf and/or anchor CSV not found"}

    maps = {
        "A": auc_by_seed(anchor_df, "A", metric),
        "A_conf": auc_by_seed(conf_df, "A_conf", metric),
        "B": auc_by_seed(anchor_df, "B", metric, sigma=ANCHOR_B_SIGMA),
        "ref": auc_by_seed(anchor_df, "ref", metric),
    }
    if any(len(m) == 0 for m in maps.values()):
        missing = [k for k, m in maps.items() if len(m) == 0]
        return {"skipped": True, "reason": f"no data for arm(s): {missing}"}

    seeds, v = aligned_vectors(maps)
    if len(seeds) < 2:
        return {"skipped": True, "reason": f"only {len(seeds)} common seed(s)"}

    mA = float(v["A"].mean())
    mAc = float(v["A_conf"].mean())
    mB = float(v["B"].mean())
    mRef = float(v["ref"].mean())

    ci_conf_gain = paired_bootstrap_ci(v["A_conf"], v["A"], seed=0)  # A_conf - A
    ci_b_vs_conf = paired_bootstrap_ci(v["B"], v["A_conf"], seed=0)  # B - A_conf
    # Recovery of B relative to the *stronger* A_conf floor.
    rec_b_over_conf = bootstrap_recovery_ci(v["A_conf"], v["B"], v["ref"], seed=0)

    return {
        "skipped": False,
        "seeds": seeds,
        "means": {"A": mA, "A_conf": mAc, "B": mB, "ref": mRef},
        "conf_gain": mAc - mA,
        "b_vs_conf": mB - mAc,
        "ci_conf_gain": (ci_conf_gain[0], ci_conf_gain[1]),
        "ci_b_vs_conf": (ci_b_vs_conf[0], ci_b_vs_conf[1]),
        "ci_recovery_b_over_conf": rec_b_over_conf,
        "conf_helps": bool(ci_conf_gain[0] > 0),
        "b_still_wins": bool(ci_b_vs_conf[0] > 0),
    }


def _md_i3(res: dict) -> list[str]:
    lines = ["## I3 - A_conf (confidence-gated self-training) vs A, and B vs A_conf", ""]
    if res.get("skipped"):
        lines += [f"_Skipped: {res['reason']}._", ""]
        return lines
    m = res["means"]
    lines += [
        f"Common seeds: {res['seeds']}.",
        "",
        "| arm | AUC |",
        "|---|---|",
        f"| A (ungated pseudo-label) | {_fmt(m['A'])} |",
        f"| A_conf (confidence gate) | {_fmt(m['A_conf'])} |",
        f"| B (simulated correction + reject) | {_fmt(m['B'])} |",
        f"| ref (full GT) | {_fmt(m['ref'])} |",
        "",
        "| comparison | delta AUC | 95% seed-bootstrap CI |",
        "|---|---|---|",
        f"| A_conf - A | {_fmt(res['conf_gain'])} | "
        f"[{_fmt(res['ci_conf_gain'][0])}, {_fmt(res['ci_conf_gain'][1])}] |",
        f"| B - A_conf | {_fmt(res['b_vs_conf'])} | "
        f"[{_fmt(res['ci_b_vs_conf'][0])}, {_fmt(res['ci_b_vs_conf'][1])}] |",
        "",
    ]
    conf = "has a CI excluding 0" if res["conf_helps"] else "has a CI including 0"
    if res["b_still_wins"]:
        comparison_note = (
            "The B-A_conf delta CI excludes 0."
        )
    else:
        comparison_note = (
            "The B-A_conf delta CI includes 0; a positive condition-B margin over the "
            "confidence-gated baseline is not established."
        )
    lines += [
        f"_Interpretation: the A_conf-A delta {conf} ({_fmt(res['conf_gain'])}). "
        f"{comparison_note}_",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_report(
    i2_df: Optional[pd.DataFrame],
    i1_df: Optional[pd.DataFrame],
    i3_df: Optional[pd.DataFrame],
    anchor_df: Optional[pd.DataFrame],
    *,
    metric: str,
) -> str:
    """Run all three analyses and assemble the markdown report."""
    lines = [
        "# Supplemental Ablation Experiments - Results",
        "",
        f"Metric: pose **{metric}** (normalized label-efficiency AUC over budgets "
        "{0.05, 0.1, 0.2, 0.4}); 95% CIs are seed-bootstrap (10k resamples over seeds).",
        "",
        "Anchor for A/ref/legacy-B: **u30** (uniform 30-epoch) run; condition B uses "
        f"the sigma={ANCHOR_B_SIGMA:g} slice.",
        "",
    ]
    if anchor_df is None:
        lines += [
            "> [!WARNING] Anchor (u30) CSV not found - every analysis needs A/ref/B "
            "from it, so all three are skipped until it lands.",
            "",
        ]
    lines += _md_i2(analyze_i2_decomp(i2_df, anchor_df, metric=metric))
    lines += _md_i1(analyze_i1_verify(i1_df, anchor_df, metric=metric))
    lines += _md_i3(analyze_i3_conf(i3_df, anchor_df, metric=metric))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--i2-csv", default=DEFAULT_I2, help="I2 filter x correct decomposition CSV")
    ap.add_argument("--i1-csv", default=DEFAULT_I1,
                    help="I1 OKS-gated selective-correction CSV")
    ap.add_argument("--i3-csv", default=DEFAULT_I3, help="I3 A_conf CSV")
    ap.add_argument("--anchor-csv", default=DEFAULT_ANCHOR, help="u30 anchor CSV (A/B/ref)")
    ap.add_argument("--metric", default="map50_95", choices=["map50_95", "map50"])
    ap.add_argument("--out-md", default=DEFAULT_OUT_MD)
    args = ap.parse_args(argv)

    i2_df = load_curve(args.i2_csv)
    i1_df = load_curve(args.i1_csv)
    i3_df = load_curve(args.i3_csv)
    anchor_df = load_curve(args.anchor_csv)

    for label, df, path in [
        ("I2 decomp", i2_df, args.i2_csv),
        ("I1 verify", i1_df, args.i1_csv),
        ("I3 A_conf", i3_df, args.i3_csv),
        ("anchor u30", anchor_df, args.anchor_csv),
    ]:
        if df is None:
            _safe_print(f"[skip] {label}: not found at {path}")

    md = build_report(i2_df, i1_df, i3_df, anchor_df, metric=args.metric)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    _safe_print(md)
    _safe_print(f"[wrote] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
