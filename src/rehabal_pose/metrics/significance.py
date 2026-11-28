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


def _validate_paired(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"paired arrays must have equal shape: {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError(f"paired arrays must be 1-D, got ndim={a.ndim}")


def paired_difference_test(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    *,
    method: str = "permutation",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
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
        n_assignments = 1 << n
        indices = np.arange(n_assignments, dtype=np.int64)
        bits = ((indices[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1)
        signs = 1 - 2 * bits
        null_stats = signs @ d / n
        p_value = float(np.mean(np.abs(null_stats) >= np.abs(t_obs) - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
        null_stats = signs @ d / n
        p_value = float(np.mean(np.abs(null_stats) >= np.abs(t_obs) - 1e-12))

    return {
        "mean_diff": t_obs,
        "statistic": t_obs,
        "p_value": p_value,
    }


def bootstrap_ci(
    values: Union[Sequence[float], np.ndarray],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
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
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    _validate_paired(a_arr, b_arr)
    return bootstrap_ci(a_arr - b_arr, alpha=alpha, n_resamples=n_resamples, seed=seed)


def bootstrap_recovery_ci(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    ref: Union[Sequence[float], np.ndarray],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
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


def compare_curves(
    df: pd.DataFrame,
    *,
    metric: str = "pose_map50_95",
    reference: str = "rehabal",
    baselines: Optional[list[str]] = None,
    group_cols: tuple[str, ...] = ("strategy", "budget", "seed"),
) -> pd.DataFrame:
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

        ref_df = budget_df[budget_df[strategy_col] == reference].sort_values(seed_col)
        ref_seeds = ref_df[seed_col].tolist()
        ref_vals = ref_df[metric].to_numpy(dtype=np.float64)

        for baseline in baselines:
            base_df = budget_df[budget_df[strategy_col] == baseline].sort_values(seed_col)
            base_seeds = base_df[seed_col].tolist()
            base_vals = base_df[metric].to_numpy(dtype=np.float64)

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

            d = ref_aligned - base_aligned
            mean_diff = float(np.mean(d))

            if np.all(d == 0.0):
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


def summarize_win_region(comparison_df: pd.DataFrame) -> str:
    required = {"budget", "baseline", "ref_better"}
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(f"comparison_df missing columns: {missing}")

    if comparison_df.empty:
        return "No comparison data available."

    budgets = sorted(comparison_df["budget"].unique().tolist())
    reference = comparison_df["reference"].iloc[0] if "reference" in comparison_df.columns else "reference"

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
