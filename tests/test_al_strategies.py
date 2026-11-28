from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rehabal_pose.al import (
    STRATEGIES,
    CoresetStrategy,
    Pool,
    RandomStrategy,
    RehabALStrategy,
    TemporalStrategy,
    UncertaintyStrategy,
    get_strategy,
    kcenter_select,
    mmr_select,
    not_implemented_infer_fn,
    not_implemented_train_fn,
    run_al_loop_step,
)

K = 26
TARGET = 8


def build_pool(n: int = 24, *, seed: int = 0) -> Pool:
    rng = np.random.default_rng(seed)
    preds = np.empty((n, K, 3), dtype=np.float64)
    preds[..., :2] = rng.uniform(0.0, 200.0, size=(n, K, 2))
    preds[..., 2] = rng.uniform(0.5, 1.0, size=(n, K))

    embeddings = rng.normal(size=(n, 6))

    frames_per_clip = 6
    clip_ids = [f"clip{i // frames_per_clip}" for i in range(n)]
    frame_ids = [i % frames_per_clip for i in range(n)]

    diseases = ["stroke", "frozen_shoulder"]
    exercises = ["ex_a", "ex_b"]
    cameras = ["camA", "camB"]
    metadata = {
        "disease": [diseases[i % 2] for i in range(n)],
        "exercise": [exercises[(i // 2) % 2] for i in range(n)],
        "camera_id": [cameras[(i // 4) % 2] for i in range(n)],
    }

    sample_ids = [f"s{i:03d}" for i in range(n)]
    return Pool(
        sample_ids=sample_ids,
        predictions=preds,
        embeddings=embeddings,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
        metadata=metadata,
    )


def clustered_pool(seed: int = 1):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    per = 8
    embs = []
    labels = []
    for c, center in enumerate(centers):
        embs.append(center + rng.normal(scale=1.0, size=(per, 2)))
        labels.extend([c] * per)
    embeddings = np.vstack(embs)
    n = embeddings.shape[0]
    preds = np.empty((n, K, 3), dtype=np.float64)
    preds[..., :2] = rng.uniform(0.0, 200.0, size=(n, K, 2))
    preds[..., 2] = rng.uniform(0.5, 1.0, size=(n, K))
    pool = Pool(
        sample_ids=[f"c{i:03d}" for i in range(n)],
        predictions=preds,
        embeddings=embeddings,
    )
    return pool, np.array(labels)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_returns_k_unique_in_range(name):
    pool = build_pool()
    strat = get_strategy(name, seed=0)
    k = 5
    selected = strat.select(pool, k)
    assert len(selected) == k
    assert len(set(selected)) == k, "indices must be unique"
    assert all(0 <= i < pool.size for i in selected)

    scores = strat.score(pool)
    assert scores.shape == (pool.size,)
    assert np.all(np.isfinite(scores))


def test_select_respects_exclude_and_budget():
    pool = build_pool()
    strat = RandomStrategy(seed=3)
    exclude = list(range(20))
    selected = strat.select(pool, 10, exclude=exclude)
    assert len(selected) == 4
    assert all(i >= 20 for i in selected)


def test_uncertainty_picks_lowest_confidence():
    pool = build_pool()
    pool.predictions[0, :, 2] = 0.05
    pool.predictions[1, :, 2] = 0.05
    pool.predictions[2, :, 2] = 0.05
    pool.predictions[3:, :, 2] = 0.95

    strat = UncertaintyStrategy()
    selected = strat.select(pool, 3)
    assert set(selected) == {0, 1, 2}


def test_uncertainty_uses_mc_variance():
    pool = build_pool(n=6)
    rng = np.random.default_rng(0)
    samples = np.repeat(pool.predictions[:, None, :, :], 5, axis=1).copy()
    samples[4, :, :, :2] += rng.normal(scale=50.0, size=(5, K, 2))
    pool = dataclasses.replace(pool, prediction_samples=samples)
    pool.predictions[:, :, 2] = 0.9

    strat = UncertaintyStrategy(variance_weight=1.0)
    scores = strat.score(pool)
    assert int(np.argmax(scores)) == 4


def test_kcenter_spreads_across_clusters():
    pool, labels = clustered_pool()
    selected = kcenter_select(pool.embeddings, 3, seed=0)
    assert len(selected) == 3
    assert len(set(labels[selected])) >= 2


def test_coreset_strategy_spreads():
    pool, labels = clustered_pool()
    strat = CoresetStrategy(seed=0)
    selected = strat.select(pool, 3)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert len(set(labels[selected])) >= 2


def test_mmr_returns_unique_in_range():
    pool, _ = clustered_pool()
    rel = np.random.default_rng(0).random(pool.size)
    selected = mmr_select(pool.embeddings, rel, 5, lambda_=0.5)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert all(0 <= i < pool.size for i in selected)


def test_rehabal_score_finite_and_in_range():
    pool = build_pool()
    strat = RehabALStrategy()
    scores = strat.score(pool)
    assert scores.shape == (pool.size,)
    assert np.all(np.isfinite(scores))
    assert scores.min() >= 0.0 - 1e-9
    assert scores.max() <= 1.0 + 1e-9


def _isolate(term: str) -> dict[str, float]:
    keys = ["uncertainty", "temporal", "angle", "rom", "compensation", "rarity"]
    return {k: (1.0 if k == term else 0.0) for k in keys}


def test_rehabal_reacts_to_uncertainty():
    pool = build_pool()
    pool.predictions[TARGET, :, 2] = 0.01
    strat = RehabALStrategy(weights=_isolate("uncertainty"))
    assert int(np.argmax(strat.score(pool))) == TARGET


def test_rehabal_reacts_to_temporal():
    pool = build_pool()
    pool.predictions[TARGET, :, :2] += 1000.0
    strat = RehabALStrategy(weights=_isolate("temporal"))
    assert int(np.argmax(strat.score(pool))) == TARGET


def test_rehabal_reacts_to_angle_sensitivity():
    pool = build_pool()
    pts = pool.predictions[TARGET, :, :2]
    centroid = pts.mean(axis=0, keepdims=True)
    pool.predictions[TARGET, :, :2] = centroid + 0.01 * (pts - centroid)
    strat = RehabALStrategy(weights=_isolate("angle"))
    assert int(np.argmax(strat.score(pool))) == TARGET


def test_rehabal_reacts_to_rom_criticality():
    from rehabal_pose.al.rehabal import JOINT_TRIPLETS
    pool = build_pool()
    P = pool.predictions[TARGET, :, :2]
    for a, b, c in JOINT_TRIPLETS:
        P[c] = P[b] + (P[b] - P[a])
    strat = RehabALStrategy(weights=_isolate("rom"))
    assert int(np.argmax(strat.score(pool))) == TARGET


def test_rehabal_reacts_to_compensation_risk():
    from rehabal_pose.al.rehabal import LR_JOINT_PAIRS
    pool = build_pool()
    P = pool.predictions[TARGET, :, :2]
    for (al, bl, cl), (ar, br, cr) in LR_JOINT_PAIRS:
        P[cr] = P[br] + (P[br] - P[ar])
        P[cl] = P[al]
    strat = RehabALStrategy(weights=_isolate("compensation"))
    assert int(np.argmax(strat.score(pool))) == TARGET


def test_rehabal_reacts_to_domain_rarity():
    pool = build_pool()
    pool.metadata["disease"][TARGET] = "ULTRA_RARE_DISEASE"
    pool.metadata["exercise"][TARGET] = "ULTRA_RARE_EX"
    pool.metadata["camera_id"][TARGET] = "ULTRA_RARE_CAM"
    strat = RehabALStrategy(weights=_isolate("rarity"))
    assert int(np.argmax(strat.score(pool))) == TARGET


def test_rehabal_select_returns_k_unique():
    pool = build_pool()
    strat = RehabALStrategy()
    selected = strat.select(pool, 6)
    assert len(selected) == 6
    assert len(set(selected)) == 6


def test_loop_retrospective_end_to_end():
    pool = build_pool()
    hidden = {sid: f"label_{i}" for i, sid in enumerate(pool.sample_ids)}
    initial_labeled = [pool.sample_ids[0], pool.sample_ids[1]]

    train_calls = {}

    def fake_train(labeled_data, labeled_ids=None):
        train_calls["n"] = len(labeled_ids or [])
        return "FAKE_MODEL"

    def fake_infer(model, p):
        assert model == "FAKE_MODEL"
        return p.predictions

    strat = UncertaintyStrategy()
    result = run_al_loop_step(
        pool,
        strat,
        k=4,
        labeled_ids=initial_labeled,
        hidden_labels=hidden,
        mode="retrospective",
        train_fn=fake_train,
        infer_fn=fake_infer,
    )

    assert result.model == "FAKE_MODEL"
    assert train_calls["n"] == 2
    assert len(result.selected_indices) == 4
    assert len(set(result.selected_indices)) == 4
    assert all(sid not in initial_labeled for sid in result.selected_sample_ids)
    assert len(result.newly_labeled) == 4
    for sid in result.selected_sample_ids:
        assert result.newly_labeled[sid] == hidden[sid]
    assert len(result.labeled_ids) == 6
    assert result.metrics["n_selected"] == 4
    assert result.metrics["n_labeled_after"] == 6
    assert np.isfinite(result.metrics["mean_selected_score"])


def test_loop_prospective_export_hook():
    pool = build_pool()
    exported_holder = {}

    def export_fn(ids, p):
        exported_holder["ids"] = list(ids)

    strat = RandomStrategy(seed=0)
    result = run_al_loop_step(
        pool,
        strat,
        k=3,
        mode="prospective",
        export_fn=export_fn,
    )
    assert result.mode == "prospective"
    assert len(result.exported) == 3
    assert exported_holder["ids"] == result.selected_sample_ids
    assert result.newly_labeled == {}


def test_loop_default_stubs_raise():
    with pytest.raises(NotImplementedError):
        not_implemented_train_fn()
    with pytest.raises(NotImplementedError):
        not_implemented_infer_fn(None, None)

    pool = build_pool(n=6)
    with pytest.raises(NotImplementedError):
        run_al_loop_step(
            pool,
            UncertaintyStrategy(),
            k=2,
            train_fn=not_implemented_train_fn,
        )
