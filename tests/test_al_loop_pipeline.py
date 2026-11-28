from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from rehabal_pose.al.loop import run_budget_curve
from rehabal_pose.metrics.learning_curve import area_under_learning_curve


def _toy_manifest(n_subjects: int = 8, clips_per_subject: int = 6) -> pd.DataFrame:
    rows = []
    for s in range(n_subjects):
        for c in range(clips_per_subject):
            sid = f"S{s:02d}_clip{c:02d}"
            rows.append(
                {
                    "sample_id": sid,
                    "subject_id": f"subj_{s:02d}",
                    "video_id": f"vid_{s:02d}_{c:02d}",
                    "exercise": f"ex{c % 3}",
                }
            )
    return pd.DataFrame(rows)


class _FakeModel:
    def __init__(self, fraction: float, n_labeled: int) -> None:
        self.fraction = fraction
        self.n_labeled = n_labeled


def _fake_train_fn(labeled_df, *, labeled_ids, seed, budget, round_idx, **_):
    pool_hint = max(len(labeled_ids), 1)
    return _FakeModel(fraction=len(labeled_ids) / pool_hint, n_labeled=len(labeled_ids))


def _fake_infer_fn(model, target_df, **_):
    n = len(target_df)
    k = 4
    preds = np.zeros((n, k, 3), dtype=np.float64)
    rng = np.random.default_rng(abs(hash(tuple(target_df["sample_id"]))) % (2**32))
    preds[..., :2] = rng.random((n, k, 2))
    nlab = getattr(model, "n_labeled", 1)
    base = 0.3 + 0.5 * rng.random((n, k))
    preds[..., 2] = np.clip(base + 0.01 * nlab, 0.0, 1.0)
    return preds


def _fake_evaluate_fn(model, test_df, predictions, *, seed, budget, n_labeled,
                      pool_size, **_):
    frac = n_labeled / max(pool_size, 1)
    acc = 0.5 + 0.45 * (1.0 - np.exp(-3.0 * frac))
    return {"accuracy": float(acc), "pck@0.05": float(acc)}


def test_no_heavy_imports_on_module_load():
    import importlib

    importlib.import_module("rehabal_pose.al.loop")
    for mod in ("torch", "ultralytics", "cv2"):
        assert mod not in sys.modules, f"{mod} must not be imported at module load"


@pytest.mark.parametrize("strategy", ["random", "uncertainty"])
def test_budget_curve_shapes_and_monotone(strategy):
    df = _toy_manifest()
    budgets = [0.1, 0.5, 1.0]
    seeds = [0, 1]
    res = run_budget_curve(
        df,
        strategy=strategy,
        budgets=budgets,
        seeds=seeds,
        train_fn=_fake_train_fn,
        infer_fn=_fake_infer_fn,
        evaluate_fn=_fake_evaluate_fn,
        metric_name="accuracy",
    )

    assert res["budgets"] == sorted(budgets)
    assert len(res["points"]) == len(budgets)
    for pt in res["points"]:
        assert "mean" in pt and "std" in pt
        assert len(pt["scores"]) == len(seeds)
        assert len(pt["n_labeled"]) == len(seeds)

    assert res["scores"].shape == (len(seeds), len(budgets))
    assert res["mean"].shape == (len(budgets),)
    assert res["std"].shape == (len(budgets),)

    assert res["monotone_increasing"] is True
    assert np.all(np.diff(res["mean"]) >= -1e-9)
    for si in range(len(seeds)):
        assert np.all(np.diff(res["scores"][si]) >= -1e-9)

    assert np.isfinite(res["auc"])
    assert np.isfinite(res["auc_normalized"])
    assert len(res["auc_per_seed"]) == len(seeds)
    expect = area_under_learning_curve(res["budgets"], res["mean"], normalize=False)
    assert res["auc"] == pytest.approx(expect)


def test_no_subject_leakage_in_splits():
    df = _toy_manifest()
    res = run_budget_curve(
        df,
        strategy="random",
        budgets=[0.2, 1.0],
        seeds=[0, 1, 2],
        train_fn=_fake_train_fn,
        infer_fn=_fake_infer_fn,
        evaluate_fn=_fake_evaluate_fn,
    )
    assert len(res["subject_splits"]) == 3
    for sp in res["subject_splits"]:
        pool_subj = set(sp["pool_subjects"])
        test_subj = set(sp["test_subjects"])
        assert pool_subj.isdisjoint(test_subj), (
            f"subject leakage seed={sp['seed']}: {pool_subj & test_subj}"
        )
        assert len(test_subj) > 0


def test_full_budget_labels_entire_pool():
    df = _toy_manifest()
    res = run_budget_curve(
        df,
        strategy="uncertainty",
        budgets=[0.1, 1.0],
        seeds=[0],
        train_fn=_fake_train_fn,
        infer_fn=_fake_infer_fn,
        evaluate_fn=_fake_evaluate_fn,
    )
    last = res["points"][-1]
    n_pool = res["subject_splits"][0]["n_pool"]
    assert last["n_labeled"][0] == n_pool
    assert res["points"][0]["n_labeled"][0] <= last["n_labeled"][0]


def test_evaluate_fn_scalar_return_supported():
    df = _toy_manifest()

    def scalar_eval(model, test_df, predictions, *, n_labeled, pool_size, **_):
        return float(n_labeled) / max(pool_size, 1)

    res = run_budget_curve(
        df,
        strategy="random",
        budgets=[0.25, 1.0],
        seeds=[0],
        train_fn=_fake_train_fn,
        infer_fn=_fake_infer_fn,
        evaluate_fn=scalar_eval,
    )
    assert np.all(np.isfinite(res["scores"]))
    assert res["scores"][0, -1] == pytest.approx(1.0)
