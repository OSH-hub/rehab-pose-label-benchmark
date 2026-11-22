"""Statistical significance testing for RehabAL-Pose learning-curve comparisons.

Provides seed-level comparison utilities for paired learning-curve analyses.
Effect sizes and confidence intervals are the primary summaries; permutation
p-values may be reported as exploratory, unadjusted supporting evidence.

All implementations are **pure NumPy** — no scipy / statsmodels / torch at
module load.  Permutation / bootstrap approaches are used throughout so the
test suite stays GPU-free.

Functions
---------
paired_difference_test   -- exact / sampled sign-flip permutation test.
bootstrap_ci             -- percentile bootstrap CI for a set of values.
paired_bootstrap_ci      -- bootstrap CI for the mean paired difference.
bootstrap_recovery_ci    -- seed-bootstrap CI for the recovery ratio
                            mean(b-a)/mean(ref-a).
compare_curves           -- tidy per-budget comparison of reference vs baselines.
summarize_win_region     -- human-readable summary of budget regions where
                            the reference strategy is significantly better.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "paired_difference_test",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "bootstrap_recovery_ci",
    "compare_curves",
    "summarize_win_region",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_paired(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"paired arrays must have equal shape: {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError(f"paired arrays must be 1-D, got ndim={a.ndim}")


# ---------------------------------------------------------------------------
# paired_difference_test
# ---------------------------------------------------------------------------


def paired_difference_test(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    *,
    method: str = "permutation",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired sign-flip permutation test for the mean difference ``a - b``.

    Parameters
    ----------
    a, b:
        Equal-length 1-D arrays of per-seed metric values (e.g. the RehabAL
        and baseline mAP scores over seeds {0, 1, 2}).
    method:
        ``"permutation"`` (default): exact test when ``n <= 12`` (all ``2^n``
        sign assignments enumerated), otherwise ``n_resamples`` random
        sign-flip assignments are drawn.  Currently the only supported method.
    n_resamples:
        Number of random assignments when the exact test is too large.
    seed:
        RNG seed for the random fallback (reproducible).

    Returns
    -------
    dict with keys:
        ``mean_diff``  -- float, ``mean(a - b)``.
        ``statistic``  -- float, same as ``mean_diff`` (the test statistic).
        ``p_value``    -- float, two-sided p-value.

    Notes
    -----
    Exact test: enumerate all ``2^n`` sign vectors ``s ∈ {-1, +1}^n`` and
    compute ``T* = mean(s * d)`` for each; ``p = P(|T*| >= |T_obs|)`` where
    ``T_obs = mean(d)`` and ``d = a - b``.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    _validate_paired(a_arr, b_arr)

    n = a_arr.size
    d = a_arr - b_arr
    t_obs = float(np.mean(d))

    if n == 0:
        return {"mean_diff": float("nan"), "statistic": float("nan"), "p_value": float("nan")}

    if method != "permutation":
        raise ValueError(f"Unsupported method: {method!r}. Use 'permutation'.")

    if n <= 12:
        # Exact: all 2^n sign assignments
        n_assignments = 1 << n  # 2^n
        # Build (2^n, n) sign matrix via bit manipulation
        indices = np.arange(n_assignments, dtype=np.int64)
        # bit k of index i gives the sign for element k: 0 -> +1, 1 -> -1
        bits = ((indices[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1)
        signs = 1 - 2 * bits  # 0->+1, 1->-1  shape (2^n, n)
        null_stats = signs @ d / n  # (2^n,) -- mean of signed differences
        p_value = float(np.mean(np.abs(null_stats) >= np.abs(t_obs) - 1e-12))
    else:
        # Random sign-flip approximation
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
        null_stats = signs @ d / n
        p_value = float(np.mean(np.abs(null_stats) >= np.abs(t_obs) - 1e-12))

    return {
        "mean_diff": t_obs,
        "statistic": t_obs,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# bootstrap_ci / paired_bootstrap_ci
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Union[Sequence[float], np.ndarray],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Parameters
    ----------
    values:
        1-D array of observed values (e.g. per-seed metric scores).
    alpha:
        Significance level; returns a ``(1 - alpha)`` CI.
    n_resamples:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (lo, hi, mean)
        Lower bound, upper bound, and point estimate (mean of ``values``).
        Returns ``(nan, nan, nan)`` for empty input.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean_val = float(np.mean(v))
    if v.size == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    resampled = rng.choice(v, size=(n_resamples, v.size), replace=True)
    boot_means = resampled.mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi, mean_val


def paired_bootstrap_ci(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean paired difference ``mean(a - b)``.

    Parameters
    ----------
    a, b:
        Equal-length 1-D paired arrays.
    alpha:
        Significance level.
    n_resamples, seed:
        Bootstrap parameters.

    Returns
    -------
    (lo, hi, mean_diff)
        Bootstrap CI bounds and the mean of ``a - b``.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    _validate_paired(a_arr, b_arr)
    return bootstrap_ci(a_arr - b_arr, alpha=alpha, n_resamples=n_resamples, seed=seed)


# ---------------------------------------------------------------------------
# bootstrap_recovery_ci
# ---------------------------------------------------------------------------


def bootstrap_recovery_ci(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    ref: Union[Sequence[float], np.ndarray],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Seed-bootstrap CI for the *recovery ratio* ``mean(b-a)/mean(ref-a)``.

    This is the descriptive recovery of the A-to-ref gap: the ratio of the mean
    B-minus-A difference to the mean ref-minus-A difference. The three
    inputs are per-seed **aligned** AUC (or metric) vectors — element ``i`` of each
    is seed ``i`` — so the same resampled seed index is applied to all three arms,
    preserving the within-seed pairing (A/B/ref share the seed's round-0 init).

    The seeds are the resampling unit: on each of ``n_resamples`` draws we resample
    seed indices with replacement, recompute ``mean(b-a)`` and ``mean(ref-a)`` on
    the resample, and take their ratio. Non-finite ratios (near-zero ceiling
    denominator, ``|mean(ref-a)| <= 1e-9``) are dropped before taking percentiles.

    Parameters
    ----------
    a, b, ref:
        Equal-length 1-D per-seed metric vectors for conditions A, B, ref.
    alpha:
        Significance level; returns a ``(1 - alpha)`` two-sided percentile CI.
    n_resamples, seed:
        Bootstrap parameters. ``seed`` fixes the RNG for a reproducible interval.

    Returns
    -------
    (lo, hi)
        Percentile CI bounds on the recovery ratio (as a fraction, e.g. 0.96 for
        96%). Returns ``(nan, nan)`` when fewer than two seeds are supplied or the
        reference gap is degenerate on every resample.

    Notes
    -----
    Mirrors the aggregate CI in ``scripts/analyze_clinician_experiment.py``
    (same RNG, resample count, and percentile convention) so the three analysis
    scripts report identical recovery intervals from the same data.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    r_arr = np.asarray(ref, dtype=np.float64).ravel()
    if not (a_arr.shape == b_arr.shape == r_arr.shape):
        raise ValueError(
            f"a/b/ref must share shape: {a_arr.shape} vs {b_arr.shape} vs {r_arr.shape}"
        )
    n = a_arr.size
    if n < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, n, size=(n_resamples, n))
    d_boot = b_arr[bidx].mean(1) - a_arr[bidx].mean(1)
    g_boot = r_arr[bidx].mean(1) - a_arr[bidx].mean(1)
    rec = np.divide(
        d_boot, g_boot,
        out=np.full_like(d_boot, np.nan),
        where=np.abs(g_boot) > 1e-9,
    )
    rec = rec[np.isfinite(rec)]
    if rec.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rec, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# compare_curves
# ---------------------------------------------------------------------------


def compare_curves(
    df: pd.DataFrame,
    *,
    metric: str = "pose_map50_95",
    reference: str = "rehabal",
    baselines: Optional[list[str]] = None,
    group_cols: tuple[str, ...] = ("strategy", "budget", "seed"),
) -> pd.DataFrame:
    """For each budget, compare the reference strategy against each baseline.

    Given a long-form ``curve.csv`` DataFrame (one row per
    ``(strategy, budget, seed)`` combination), this function groups by
    ``(strategy, budget)`` to obtain per-seed metric vectors, then calls
    :func:`paired_difference_test` and :func:`paired_bootstrap_ci` to produce
    a tidy result table.

    Parameters
    ----------
    df:
        Long-form DataFrame.  Must contain ``group_cols`` + ``metric`` columns.
    metric:
        Name of the metric column to compare (default ``"pose_map50_95"``).
    reference:
        Strategy name used as the *reference* (e.g. ``"rehabal"``).
    baselines:
        List of baseline strategy names to compare against.  If ``None`` all
        strategies other than ``reference`` are used (sorted for
        determinism).
    group_cols:
        Column names for (strategy, budget, seed) — in that order.

    Returns
    -------
    pandas.DataFrame with columns:
        ``budget, reference, baseline, mean_ref, mean_base, mean_diff,
        ci_lo, ci_hi, p_value, ref_better``

        ``ref_better`` is ``True`` when ``p_value < 0.05`` AND
        ``mean_diff > 0``.

    Notes
    -----
    At the 100 % budget point (or any budget where all strategies yield
    identical values / zero variance) the permutation test handles the
    degenerate case gracefully: all sign-flip statistics equal the observed
    statistic, so ``p_value = 1.0`` and ``ref_better = False``.
    """
    strategy_col, budget_col, seed_col = group_cols

    required_cols = {strategy_col, budget_col, seed_col, metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    strategies = df[strategy_col].unique().tolist()
    if reference not in strategies:
        raise ValueError(f"Reference strategy {reference!r} not found in DataFrame.")

    if baselines is None:
        baselines = sorted(s for s in strategies if s != reference)
    else:
        baselines = list(baselines)

    budgets = sorted(df[budget_col].unique().tolist())

    rows: list[dict] = []

    for budget in budgets:
        budget_df = df[df[budget_col] == budget]

        # Build seed -> metric value mapping for the reference strategy
        ref_df = budget_df[budget_df[strategy_col] == reference].sort_values(seed_col)
        ref_seeds = ref_df[seed_col].tolist()
        ref_vals = ref_df[metric].to_numpy(dtype=np.float64)

        for baseline in baselines:
            base_df = budget_df[budget_df[strategy_col] == baseline].sort_values(seed_col)
            base_seeds = base_df[seed_col].tolist()
            base_vals = base_df[metric].to_numpy(dtype=np.float64)

            # Align on seeds that appear in both (handles missing seeds)
            common_seeds = sorted(set(ref_seeds) & set(base_seeds))
            if len(common_seeds) == 0:
                rows.append({
                    "budget": budget,
                    "reference": reference,
                    "baseline": baseline,
                    "mean_ref": float("nan"),
                    "mean_base": float("nan"),
                    "mean_diff": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "p_value": float("nan"),
                    "ref_better": False,
                })
                continue

            ref_aligned = np.array(
                [ref_df.loc[ref_df[seed_col] == s, metric].iloc[0] for s in common_seeds],
                dtype=np.float64,
            )
            base_aligned = np.array(
                [base_df.loc[base_df[seed_col] == s, metric].iloc[0] for s in common_seeds],
                dtype=np.float64,
            )

            # Degenerate case: zero variance / all equal -> p=1.0
            d = ref_aligned - base_aligned
            mean_diff = float(np.mean(d))

            if np.all(d == 0.0):
                # All seeds agree; test is vacuous
                ci_lo, ci_hi, _ = float(mean_diff), float(mean_diff), mean_diff
                p_value = 1.0
            else:
                test_result = paired_difference_test(ref_aligned, base_aligned, seed=0)
                p_value = test_result["p_value"]
                ci_lo, ci_hi, _ = paired_bootstrap_ci(
                    ref_aligned, base_aligned, seed=0
                )

            rows.append({
                "budget": budget,
                "reference": reference,
                "baseline": baseline,
                "mean_ref": float(np.mean(ref_aligned)),
                "mean_base": float(np.mean(base_aligned)),
                "mean_diff": mean_diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_value": p_value,
                "ref_better": bool(p_value < 0.05 and mean_diff > 0),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# summarize_win_region
# ---------------------------------------------------------------------------


def summarize_win_region(comparison_df: pd.DataFrame) -> str:
    """Summarize budget regions where nominal comparisons favor the reference.

    The summary reports budgets at which the reference is nominally better than
    all baselines, some baselines, or none under the supplied comparisons.

    Parameters
    ----------
    comparison_df:
        Output of :func:`compare_curves`.  Must contain columns
        ``budget``, ``baseline``, ``ref_better``.

    Returns
    -------
    str
        A short, deterministic summary paragraph.
    """
    required = {"budget", "baseline", "ref_better"}
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(f"comparison_df missing columns: {missing}")

    if comparison_df.empty:
        return "No comparison data available."

    budgets = sorted(comparison_df["budget"].unique().tolist())
    reference = comparison_df["reference"].iloc[0] if "reference" in comparison_df.columns else "reference"

    # Per budget: how many nominal, unadjusted tests favor the reference?
    all_budgets: list = []
    some_budgets: list = []
    none_budgets: list = []

    for budget in budgets:
        sub = comparison_df[comparison_df["budget"] == budget]
        n_baselines = len(sub)
        n_better = int(sub["ref_better"].sum())

        if n_baselines == 0:
            none_budgets.append(budget)
        elif n_better == n_baselines:
            all_budgets.append(budget)
        elif n_better > 0:
            some_budgets.append(budget)
        else:
            none_budgets.append(budget)

    def _fmt(blist: list) -> str:
        return ", ".join(str(b) for b in sorted(blist)) if blist else "none"

    parts: list[str] = [
        f"Exploratory characterization of '{reference}' vs baselines across seeds:"
    ]

    if all_budgets:
        parts.append(
            f"  Nominally better than ALL baselines at budget(s): {_fmt(all_budgets)}."
        )
    if some_budgets:
        parts.append(
            f"  Nominally better than SOME (but not all) baselines at budget(s): {_fmt(some_budgets)}."
        )
    if none_budgets:
        parts.append(
            f"  No nominal advantage at budget(s): {_fmt(none_budgets)}."
        )

    if not all_budgets and not some_budgets:
        parts.append("  Conclusion: no nominal advantage at any tested budget.")
    elif all_budgets and not some_budgets and not none_budgets:
        parts.append("  Conclusion: nominally better at ALL tested budgets.")
    else:
        parts.append(
            "  Conclusion: nominal results vary by budget; inspect effect sizes "
            "and confidence intervals rather than this unadjusted classification."
        )

    return "\n".join(parts)
