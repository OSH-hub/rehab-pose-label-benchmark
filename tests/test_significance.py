from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rehabal_pose.metrics.significance import (
    bootstrap_ci,
    bootstrap_recovery_ci,
    compare_curves,
    paired_bootstrap_ci,
    paired_difference_test,
    summarize_win_region,
)


def _make_curve_df(
    strategies: dict[str, dict[int, list[float]]],
    metric: str = "pose_map50_95",
    seeds: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    rows = []
    for strat, budget_map in strategies.items():
        for budget, scores in budget_map.items():
            for seed_idx, score in zip(seeds, scores):
                rows.append(
                    {
                        "strategy": strat,
                        "budget": budget,
                        "seed": seed_idx,
                        metric: score,
                    }
                )
    return pd.DataFrame(rows)


class TestPairedDifferenceTest:
    def test_clear_positive_difference(self):
        a = np.array([0.9, 0.85, 0.88])
        b = np.array([0.5, 0.48, 0.52])
        result = paired_difference_test(a, b)
        assert result["mean_diff"] == pytest.approx(float(np.mean(a - b)), rel=1e-9)
        assert result["p_value"] <= 0.25

    def test_zero_difference_p_equals_one(self):
        a = np.array([0.7, 0.7, 0.7])
        b = np.array([0.7, 0.7, 0.7])
        result = paired_difference_test(a, b)
        assert result["mean_diff"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0)

    def test_two_sided_symmetry(self):
        a = np.array([0.9, 0.85, 0.88])
        b = np.array([0.5, 0.48, 0.52])
        r_ab = paired_difference_test(a, b)
        r_ba = paired_difference_test(b, a)
        assert r_ab["p_value"] == pytest.approx(r_ba["p_value"], rel=1e-9)
        assert r_ab["mean_diff"] == pytest.approx(-r_ba["mean_diff"], rel=1e-9)

    def test_exact_n3_all_same_sign(self):
        a = np.array([0.9, 0.85, 0.88])
        b = np.array([0.5, 0.48, 0.52])
        result = paired_difference_test(a, b)
        assert result["p_value"] == pytest.approx(2 / 8, abs=1e-10)

    def test_returns_required_keys(self):
        result = paired_difference_test([0.8, 0.75], [0.6, 0.65])
        assert set(result.keys()) == {"mean_diff", "statistic", "p_value"}
        assert result["statistic"] == result["mean_diff"]

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            paired_difference_test([0.8], [0.6], method="ttest")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal shape"):
            paired_difference_test([0.8, 0.7], [0.6])

    def test_large_n_uses_sampling(self):
        rng = np.random.default_rng(42)
        a = rng.uniform(0.8, 0.9, size=15)
        b = rng.uniform(0.4, 0.6, size=15)
        result = paired_difference_test(a, b, n_resamples=5000, seed=0)
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["p_value"] < 0.05

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(7)
        a = rng.uniform(0.6, 0.7, size=15)
        b = rng.uniform(0.4, 0.5, size=15)
        r1 = paired_difference_test(a, b, n_resamples=1000, seed=42)
        r2 = paired_difference_test(a, b, n_resamples=1000, seed=42)
        assert r1["p_value"] == r2["p_value"]


class TestBootstrapCI:
    def test_ci_contains_mean(self):
        values = [0.7, 0.75, 0.72]
        lo, hi, mean = bootstrap_ci(values, seed=0)
        assert lo <= mean <= hi

    def test_mean_value(self):
        values = [0.7, 0.75, 0.72]
        _, _, mean = bootstrap_ci(values, seed=0)
        assert mean == pytest.approx(float(np.mean(values)), rel=1e-9)

    def test_empty_returns_nan(self):
        lo, hi, mean = bootstrap_ci([], seed=0)
        assert np.isnan(lo) and np.isnan(hi) and np.isnan(mean)

    def test_single_value_returns_that_value(self):
        lo, hi, mean = bootstrap_ci([0.8], seed=0)
        assert lo == pytest.approx(0.8)
        assert hi == pytest.approx(0.8)
        assert mean == pytest.approx(0.8)

    def test_wider_ci_with_higher_variance(self):
        low_var = [0.70, 0.71, 0.70, 0.71, 0.70] * 100
        high_var = [0.20, 0.90, 0.20, 0.90, 0.20] * 100
        lo_lv, hi_lv, _ = bootstrap_ci(low_var, seed=0)
        lo_hv, hi_hv, _ = bootstrap_ci(high_var, seed=0)
        assert (hi_hv - lo_hv) > (hi_lv - lo_lv)

    def test_deterministic(self):
        v = [0.7, 0.75, 0.72, 0.68]
        r1 = bootstrap_ci(v, seed=123)
        r2 = bootstrap_ci(v, seed=123)
        assert r1 == r2


class TestPairedBootstrapCI:
    def test_ci_around_mean_diff(self):
        a = [0.9, 0.85, 0.88]
        b = [0.5, 0.48, 0.52]
        lo, hi, mean = paired_bootstrap_ci(a, b, seed=0)
        expected_mean = float(np.mean(np.array(a) - np.array(b)))
        assert mean == pytest.approx(expected_mean, rel=1e-9)
        assert lo <= mean <= hi

    def test_zero_diff_ci_at_zero(self):
        a = [0.7, 0.7, 0.7]
        b = [0.7, 0.7, 0.7]
        lo, hi, mean = paired_bootstrap_ci(a, b, seed=0)
        assert mean == pytest.approx(0.0)

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci([0.8, 0.7], [0.6])


class TestCompareCurves:
    @pytest.fixture
    def curve_df(self):
        strategies = {
            "rehabal": {
                10: [0.90, 0.88, 0.91],
                100: [0.95, 0.95, 0.95],
            },
            "random": {
                10: [0.50, 0.48, 0.52],
                100: [0.95, 0.95, 0.95],
            },
            "uncertainty": {
                10: [0.55, 0.53, 0.57],
                100: [0.95, 0.95, 0.95],
            },
        }
        return _make_curve_df(strategies)

    def test_output_columns(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        expected_cols = {
            "budget", "reference", "baseline", "mean_ref", "mean_base",
            "mean_diff", "ci_lo", "ci_hi", "p_value", "ref_better",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_ref_better_at_low_budget_direction(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        low_budget = result[result["budget"] == 10]
        assert (low_budget["mean_diff"] > 0).all(), "Expected positive mean_diff at budget=10"
        p_vals = low_budget["p_value"].to_numpy(dtype=float)
        assert np.allclose(p_vals, 0.25), f"Expected p=0.25 (min for n=3), got {p_vals}"

    def test_no_advantage_at_full_budget(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        full_budget = result[result["budget"] == 100]
        assert not full_budget["ref_better"].any(), (
            f"Expected ref_better=False for all baselines at budget=100:\n{full_budget}"
        )

    def test_p_value_lowest_at_low_budget(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        for baseline in ["random", "uncertainty"]:
            p_low = result.loc[
                (result["budget"] == 10) & (result["baseline"] == baseline), "p_value"
            ].iloc[0]
            p_full = result.loc[
                (result["budget"] == 100) & (result["baseline"] == baseline), "p_value"
            ].iloc[0]
            assert p_low < p_full, f"p_value at budget=10 should be < at budget=100 for {baseline}"

    def test_p_value_high_at_full_budget(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        full_budget = result[result["budget"] == 100]
        for p in full_budget["p_value"]:
            assert p == pytest.approx(1.0)

    def test_mean_values_correct(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        low_random = result[
            (result["budget"] == 10) & (result["baseline"] == "random")
        ].iloc[0]
        assert low_random["mean_ref"] == pytest.approx(np.mean([0.90, 0.88, 0.91]), rel=1e-6)
        assert low_random["mean_base"] == pytest.approx(np.mean([0.50, 0.48, 0.52]), rel=1e-6)

    def test_explicit_baselines_subset(self, curve_df):
        result = compare_curves(
            curve_df, reference="rehabal", baselines=["random"]
        )
        assert set(result["baseline"].unique()) == {"random"}

    def test_missing_reference_raises(self, curve_df):
        with pytest.raises(ValueError, match="not found"):
            compare_curves(curve_df, reference="nonexistent")

    def test_missing_column_raises(self, curve_df):
        bad_df = curve_df.drop(columns=["pose_map50_95"])
        with pytest.raises(ValueError, match="missing columns"):
            compare_curves(bad_df, reference="rehabal")

    def test_tidy_row_count(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        assert len(result) == 4

    def test_ref_better_boolean_dtype(self, curve_df):
        result = compare_curves(curve_df, reference="rehabal")
        assert result["ref_better"].dtype == bool or result["ref_better"].dtype == object
        for v in result["ref_better"]:
            assert isinstance(v, (bool, np.bool_))


class TestSummarizeWinRegion:
    @pytest.fixture
    def comparison_df(self):
        strategies = {
            "rehabal": {
                10: [0.90, 0.88, 0.91],
                50: [0.80, 0.78, 0.82],
                100: [0.95, 0.95, 0.95],
            },
            "random": {
                10: [0.50, 0.48, 0.52],
                50: [0.78, 0.77, 0.79],
                100: [0.95, 0.95, 0.95],
            },
        }
        df = _make_curve_df(strategies)
        return compare_curves(df, reference="rehabal", baselines=["random"])

    def test_returns_string(self, comparison_df):
        summary = summarize_win_region(comparison_df)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_mentions_reference_name(self, comparison_df):
        summary = summarize_win_region(comparison_df)
        assert "rehabal" in summary

    def test_win_at_low_budget_mentioned(self, comparison_df):
        summary = summarize_win_region(comparison_df)
        assert "10" in summary

    def test_no_advantage_budget_mentioned(self, comparison_df):
        summary = summarize_win_region(comparison_df)
        assert "100" in summary

    def test_deterministic(self, comparison_df):
        s1 = summarize_win_region(comparison_df)
        s2 = summarize_win_region(comparison_df)
        assert s1 == s2

    def test_empty_df_returns_message(self):
        empty = pd.DataFrame(columns=["budget", "baseline", "ref_better"])
        result = summarize_win_region(empty)
        assert "No comparison data" in result

    def test_missing_columns_raises(self):
        bad = pd.DataFrame({"budget": [10], "baseline": ["random"]})
        with pytest.raises(ValueError, match="missing columns"):
            summarize_win_region(bad)

    def test_all_better(self):
        df = pd.DataFrame({
            "budget": [10, 20],
            "reference": ["rehabal", "rehabal"],
            "baseline": ["random", "random"],
            "mean_ref": [0.9, 0.85],
            "mean_base": [0.5, 0.55],
            "mean_diff": [0.4, 0.3],
            "ci_lo": [0.3, 0.2],
            "ci_hi": [0.5, 0.4],
            "p_value": [0.01, 0.02],
            "ref_better": [True, True],
        })
        summary = summarize_win_region(df)
        assert "ALL tested budgets" in summary or "ALL baselines" in summary

    def test_none_better(self):
        df = pd.DataFrame({
            "budget": [10, 20],
            "reference": ["rehabal", "rehabal"],
            "baseline": ["random", "random"],
            "mean_ref": [0.7, 0.7],
            "mean_base": [0.7, 0.7],
            "mean_diff": [0.0, 0.0],
            "ci_lo": [0.0, 0.0],
            "ci_hi": [0.0, 0.0],
            "p_value": [1.0, 1.0],
            "ref_better": [False, False],
        })
        summary = summarize_win_region(df)
        assert "no nominal advantage" in summary


class TestBootstrapRecoveryCI:
    def test_perfect_recovery_brackets_one(self):
        a = [0.20, 0.22, 0.18, 0.21, 0.19, 0.20]
        ref = [0.46, 0.47, 0.45, 0.46, 0.47, 0.46]
        b = list(ref)
        lo, hi = bootstrap_recovery_ci(a, b, ref)
        assert lo == pytest.approx(1.0, abs=1e-9)
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_half_recovery_between_zero_and_one(self):
        a = np.array([0.20, 0.22, 0.18, 0.21, 0.19, 0.20])
        ref = np.array([0.46, 0.47, 0.45, 0.46, 0.47, 0.46])
        b = a + 0.5 * (ref - a)
        lo, hi = bootstrap_recovery_ci(a, b, ref)
        assert 0.0 < lo <= 0.5 <= hi < 1.0

    def test_reproducible_fixed_seed(self):
        a = [0.20, 0.22, 0.18, 0.21, 0.19, 0.20]
        b = [0.44, 0.45, 0.42, 0.46, 0.41, 0.45]
        ref = [0.46, 0.47, 0.45, 0.46, 0.47, 0.46]
        assert bootstrap_recovery_ci(a, b, ref) == bootstrap_recovery_ci(a, b, ref)

    def test_too_few_seeds_returns_nan(self):
        lo, hi = bootstrap_recovery_ci([0.2], [0.4], [0.45])
        assert np.isnan(lo) and np.isnan(hi)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            bootstrap_recovery_ci([0.2, 0.3], [0.4], [0.45, 0.5])
