"""Tests for scripts/analyze_review_experiments.py on synthetic 6-seed curves.

All fixtures are pure in-memory DataFrames matching the clinician_curve.csv
schema (no GPU, no real data). We check:

* numeric consistency of the I2 2x2 decomposition (filter + correct + interaction
  == total gain, and interaction matches the closed-form cross term),
* I1 selective-correction recovery arithmetic and the comparison with A,
* I3 A_conf gain and the B-versus-A_conf comparison,
* graceful skipping on missing / partial inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import analyze_review_experiments as ar  # noqa: E402

BUDGETS = [0.05, 0.1, 0.2, 0.4]
# Normalized trapezoidal AUC of the linear curve f(b)=level+0.5*b equals the
# exact average of a line over [b_min, b_max]: level + 0.5*(b_min+b_max)/2.
AUC_OFFSET = 0.5 * (BUDGETS[0] + BUDGETS[-1]) / 2  # = 0.1125
SEEDS = [0, 1, 2, 3, 4, 5]
COLS = [
    "condition", "seed", "sigma", "budget", "n_labeled",
    "map50", "map50_95", "n_proposed", "n_accepted", "n_rejected",
    "n_corrected", "clinician",
]


def _rows(condition, level, *, sigma="", jitter=0.0, seeds=SEEDS):
    """Build curve rows whose map50_95 rises with budget, offset by `level`.

    Each seed gets a small deterministic per-seed jitter so bootstrap CIs are
    non-degenerate but the mean is controlled by `level`.
    """
    out = []
    for s in seeds:
        seed_off = (s - 2.5) * jitter
        for b in BUDGETS:
            val = level + 0.5 * b + seed_off
            out.append({
                "condition": condition, "seed": s, "sigma": sigma, "budget": b,
                "n_labeled": int(b * 1000), "map50": min(1.0, val + 0.2),
                "map50_95": val, "n_proposed": 100, "n_accepted": 50,
                "n_rejected": 50, "n_corrected": 50, "clinician": "simulated",
            })
    return out


def _df(rows):
    return pd.DataFrame(rows, columns=COLS)


# ---------------------------------------------------------------------------
# Anchor fixture: A / B(sigma=0 and a decoy sigma=5) / ref
# ---------------------------------------------------------------------------


@pytest.fixture
def anchor_df():
    rows = []
    rows += _rows("A", 0.30, jitter=0.01)
    rows += _rows("ref", 0.50, jitter=0.01)
    rows += _rows("B", 0.45, sigma=0.0, jitter=0.01)      # canonical slice
    rows += _rows("B", 0.10, sigma=5.0, jitter=0.01)      # decoy: must be filtered out
    return _df(rows)


# ---------------------------------------------------------------------------
# I2 decomposition
# ---------------------------------------------------------------------------


@pytest.fixture
def i2_df():
    rows = []
    rows += _rows("A_filter", 0.36, jitter=0.01)         # filter only
    rows += _rows("B_correct_only", 0.40, jitter=0.01)   # correct only
    rows += _rows("ref_filter", 0.45, jitter=0.01)       # both (== B)
    rows += _rows("B_reject_only", 0.34, jitter=0.01)    # supplementary
    return _df(rows)


def test_i2_decomposition_closes(anchor_df, i2_df):
    res = ar.analyze_i2_decomp(i2_df, anchor_df, metric="map50_95")
    assert not res["skipped"]
    # filter + correct + interaction == total gain (exact identity)
    recomposed = res["filtering"] + res["correction"] + res["interaction"]
    assert res["total_gain"] == pytest.approx(recomposed, abs=1e-12)
    # interaction == closed-form cross term mB - mAf - mBc + mA
    m = res["means"]
    cross = m["B"] - m["A_filter"] - m["B_correct_only"] + m["A"]
    assert res["interaction"] == pytest.approx(cross, abs=1e-12)
    # levels chosen so total gain = 0.45-0.30 = 0.15 (level diff; budget term cancels)
    assert res["total_gain"] == pytest.approx(0.15, abs=1e-9)
    assert res["filtering"] == pytest.approx(0.06, abs=1e-9)     # 0.36-0.30
    assert res["correction"] == pytest.approx(0.10, abs=1e-9)    # 0.40-0.30
    assert res["interaction"] == pytest.approx(-0.01, abs=1e-9)  # 0.15-0.06-0.10
    # reject-only supplementary picked up
    assert res["reject_only_mean"] == pytest.approx(0.34 + AUC_OFFSET, abs=1e-9)


def test_i2_markdown_renders(anchor_df, i2_df):
    res = ar.analyze_i2_decomp(i2_df, anchor_df, metric="map50_95")
    md = "\n".join(ar._md_i2(res))
    assert "2x2 decomposition" in md
    assert "total gain" in md


# ---------------------------------------------------------------------------
# I1 verify
# ---------------------------------------------------------------------------


@pytest.fixture
def i1_df():
    # selective-correction B sits between A(0.30) and ref(0.50), below always-correct B(0.45)
    return _df(_rows("B", 0.42, jitter=0.01))


def test_i1_recovery_arithmetic(anchor_df, i1_df):
    res = ar.analyze_i1_verify(i1_df, anchor_df, metric="map50_95")
    assert not res["skipped"]
    # gap = ref - A = 0.20; recovery_legacy = (0.45-0.30)/0.20 = 0.75
    assert res["gap"] == pytest.approx(0.20, abs=1e-9)
    assert res["recovery_legacy"] == pytest.approx(0.75, abs=1e-9)
    # selective-correction recovery = (0.42-0.30)/0.20 = 0.60
    assert res["recovery_verify"] == pytest.approx(0.60, abs=1e-9)
    assert res["verify_beats_a"] is True  # +0.12 delta, tight jitter -> CI excludes 0


def test_i1_verdict_in_markdown(anchor_df, i1_df):
    res = ar.analyze_i1_verify(i1_df, anchor_df, metric="map50_95")
    md = "\n".join(ar._md_i1(res))
    assert "selective correction" in md.lower()
    assert "recovery" in md.lower()


def test_i1_ignores_decoy_sigma(anchor_df, i1_df):
    # always-correct B must use sigma=0 (0.45), not the sigma=5 decoy (0.10).
    res = ar.analyze_i1_verify(i1_df, anchor_df, metric="map50_95")
    assert res["means"]["B_legacy"] == pytest.approx(0.45 + AUC_OFFSET, abs=1e-9)


# ---------------------------------------------------------------------------
# I3 A_conf
# ---------------------------------------------------------------------------


@pytest.fixture
def i3_df():
    return _df(_rows("A_conf", 0.35, jitter=0.01))  # between A(0.30) and B(0.45)


def test_i3_gains(anchor_df, i3_df):
    res = ar.analyze_i3_conf(i3_df, anchor_df, metric="map50_95")
    assert not res["skipped"]
    assert res["conf_gain"] == pytest.approx(0.05, abs=1e-9)   # 0.35-0.30
    assert res["b_vs_conf"] == pytest.approx(0.10, abs=1e-9)   # 0.45-0.35
    assert res["conf_helps"] is True
    assert res["b_still_wins"] is True


def test_i3_b_loses_when_conf_strong(anchor_df):
    # A_conf as strong as B -> clinician margin should NOT be established.
    strong = _df(_rows("A_conf", 0.45, jitter=0.01))
    res = ar.analyze_i3_conf(strong, anchor_df, metric="map50_95")
    assert res["b_vs_conf"] == pytest.approx(0.0, abs=1e-9)
    assert res["b_still_wins"] is False


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_missing_anchor_skips_all():
    for fn in (ar.analyze_i2_decomp, ar.analyze_i1_verify, ar.analyze_i3_conf):
        res = fn(_df(_rows("A_filter", 0.3)), None, metric="map50_95")
        assert res["skipped"]


def test_missing_experiment_skips(anchor_df):
    res = ar.analyze_i2_decomp(None, anchor_df, metric="map50_95")
    assert res["skipped"]


def test_partial_condition_skips(anchor_df):
    # decomp CSV missing B_correct_only -> skip, don't crash
    partial = _df(_rows("A_filter", 0.36) + _rows("ref_filter", 0.45))
    res = ar.analyze_i2_decomp(partial, anchor_df, metric="map50_95")
    assert res["skipped"]
    assert "B_correct_only" in res["reason"]


def test_build_report_partial_runs(anchor_df, i2_df):
    # Only I2 available; I1/I3 missing -> report still builds with skip notes.
    md = ar.build_report(i2_df, None, None, anchor_df, metric="map50_95")
    assert "I2 - Filtering" in md
    assert "Skipped" in md  # I1 and I3 skip notes present
    assert md.endswith("\n")


def test_single_seed_skips(i2_df):
    one = _df(_rows("A", 0.30, seeds=[0]) + _rows("ref", 0.50, seeds=[0])
              + _rows("B", 0.45, sigma=0.0, seeds=[0]))
    res = ar.analyze_i2_decomp(i2_df, one, metric="map50_95")
    assert res["skipped"]
