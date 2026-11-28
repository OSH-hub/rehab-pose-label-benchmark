from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .base import AcquisitionStrategy, Pool

__all__ = [
    "AlStepResult",
    "run_al_loop_step",
    "run_budget_curve",
    "not_implemented_train_fn",
    "not_implemented_infer_fn",
]


def not_implemented_train_fn(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "No training backend is configured. Inject a `train_fn` (callable "
        "returning a model) into run_al_loop_step; training "
        "requires an installed model backend and authorized dataset access."
    )


def not_implemented_infer_fn(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        "No inference backend is configured. Inject an `infer_fn` (callable "
        "returning predictions or an updated Pool) into "
        "run_al_loop_step."
    )


@dataclass
class AlStepResult:
    selected_indices: list[int]
    selected_sample_ids: list[str]
    scores: np.ndarray
    labeled_ids: set[str]
    newly_labeled: dict[str, Any] = field(default_factory=dict)
    exported: list[str] = field(default_factory=list)
    mode: str = "retrospective"
    metrics: dict[str, Any] = field(default_factory=dict)
    model: Any = None


def run_al_loop_step(
    pool: Pool,
    strategy: AcquisitionStrategy,
    k: int,
    *,
    labeled_ids: Optional[Sequence[str]] = None,
    hidden_labels: Optional[dict[str, Any]] = None,
    mode: str = "retrospective",
    train_fn: Optional[Callable[..., Any]] = None,
    infer_fn: Optional[Callable[..., Any]] = None,
    export_fn: Optional[Callable[[list[str], Pool], Any]] = None,
    labeled_data: Any = None,
    model: Any = None,
    metrics_fn: Optional[Callable[[AlStepResult], dict[str, Any]]] = None,
) -> AlStepResult:
    if mode not in ("retrospective", "prospective"):
        raise ValueError(f"mode must be 'retrospective'|'prospective'; got {mode!r}")

    labeled = set(labeled_ids or [])
    hidden = dict(hidden_labels or {})

    if train_fn is not None:
        model = train_fn(labeled_data, labeled_ids=labeled)

    working_pool = pool
    if infer_fn is not None:
        inferred = infer_fn(model, working_pool)
        if isinstance(inferred, Pool):
            working_pool = inferred
        elif inferred is not None:
            preds = np.asarray(inferred, dtype=np.float64)
            working_pool = dataclasses.replace(working_pool, predictions=preds)

    exclude = [
        i for i, sid in enumerate(working_pool.sample_ids) if sid in labeled
    ]

    scores = np.asarray(strategy.score(working_pool), dtype=np.float64)
    selected = strategy.select(working_pool, k, exclude=exclude)
    selected_ids = [working_pool.sample_ids[i] for i in selected]

    newly_labeled: dict[str, Any] = {}
    exported: list[str] = []
    if mode == "retrospective":
        for sid in selected_ids:
            newly_labeled[sid] = hidden.get(sid)
        labeled |= set(selected_ids)
    else:
        exported = list(selected_ids)
        if export_fn is not None:
            export_fn(selected_ids, working_pool)

    sel_scores = scores[selected] if selected else np.array([], dtype=np.float64)
    metrics: dict[str, Any] = {
        "n_pool": working_pool.size,
        "n_labeled_before": len(set(labeled_ids or [])),
        "n_selected": len(selected),
        "n_labeled_after": len(labeled),
        "n_revealed": len(newly_labeled),
        "mean_selected_score": float(sel_scores.mean()) if sel_scores.size else 0.0,
        "max_selected_score": float(sel_scores.max()) if sel_scores.size else 0.0,
        "mean_pool_score": float(scores.mean()) if scores.size else 0.0,
        "mode": mode,
        "strategy": getattr(strategy, "name", type(strategy).__name__),
    }

    result = AlStepResult(
        selected_indices=list(selected),
        selected_sample_ids=selected_ids,
        scores=scores,
        labeled_ids=labeled,
        newly_labeled=newly_labeled,
        exported=exported,
        mode=mode,
        metrics=metrics,
        model=model,
    )

    if metrics_fn is not None:
        result.metrics.update(metrics_fn(result))

    return result


def _resolve_strategy(
    strategy: Any, seed: int, strategy_kwargs: dict[str, Any]
) -> AcquisitionStrategy:
    if isinstance(strategy, AcquisitionStrategy):
        return strategy
    if isinstance(strategy, str):
        from . import get_strategy

        kwargs = dict(strategy_kwargs)
        kwargs.setdefault("seed", seed)
        return get_strategy(strategy, **kwargs)
    if callable(strategy):
        return strategy(seed)
    raise TypeError(
        "strategy must be a name, an AcquisitionStrategy, or a factory "
        f"callable; got {type(strategy).__name__}"
    )


def _predictions_for_df(
    infer_out: Any,
    sample_ids: Sequence[str],
    *,
    num_keypoints: Optional[int] = None,
) -> Optional[np.ndarray]:
    if infer_out is None:
        return None
    ids = [str(s) for s in sample_ids]
    n = len(ids)

    if isinstance(infer_out, Pool):
        return infer_out.predictions

    if isinstance(infer_out, np.ndarray):
        arr = np.asarray(infer_out, dtype=np.float64)
        return arr if arr.ndim == 3 else None

    by_id: dict[str, np.ndarray] = {}
    if isinstance(infer_out, dict):
        for sid, kp in infer_out.items():
            by_id[str(sid)] = np.asarray(kp, dtype=np.float64)
    else:
        try:
            iterator = list(infer_out)
        except TypeError:
            return None
        for item in iterator:
            sid = getattr(item, "sample_id", None)
            kp = getattr(item, "keypoints", None)
            if sid is None or kp is None:
                return None
            by_id[str(sid)] = np.asarray(kp, dtype=np.float64)

    if not by_id:
        return None
    k = num_keypoints
    if k is None:
        k = int(next(iter(by_id.values())).shape[0])
    out = np.zeros((n, k, 3), dtype=np.float64)
    for i, sid in enumerate(ids):
        kp = by_id.get(sid)
        if kp is not None and kp.shape[0] == k:
            out[i, :, : kp.shape[1]] = kp[:, :3]
    return out


def _build_pool(
    df: Any,
    predictions: Optional[np.ndarray],
    embeddings: Optional[np.ndarray],
    *,
    sample_id_col: str,
    clip_col: str = "video_id",
    frame_col: str = "frame_id",
) -> Pool:
    ids = [str(s) for s in df[sample_id_col].tolist()]
    clip_ids = None
    if clip_col in getattr(df, "columns", []):
        clip_ids = [c for c in df[clip_col].tolist()]
    frame_ids = None
    if frame_col in getattr(df, "columns", []):
        try:
            frame_ids = [int(f) for f in df[frame_col].tolist()]
        except (TypeError, ValueError):
            frame_ids = None
    return Pool(
        sample_ids=ids,
        predictions=predictions,
        embeddings=embeddings,
        clip_ids=clip_ids,
        frame_ids=frame_ids,
    )


def _metric_scalar(metric: Any, metric_name: str) -> float:
    if isinstance(metric, dict):
        if metric_name in metric:
            return float(metric[metric_name])
        for v in metric.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return float("nan")
    return float(metric)


def run_budget_curve(
    manifest_df: Any,
    *,
    strategy: Any,
    budgets: Sequence[float],
    seeds: Sequence[int],
    train_fn: Callable[..., Any],
    infer_fn: Callable[..., Any],
    evaluate_fn: Callable[..., Any],
    initial_fraction: float = 0.0,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    group_col: str = "subject_id",
    sample_id_col: str = "sample_id",
    pool_splits: Sequence[str] = ("train", "val"),
    test_split: str = "test",
    metric_name: str = "accuracy",
    num_keypoints: Optional[int] = None,
    strategy_kwargs: Optional[dict[str, Any]] = None,
    init_strategy: str = "random",
    embed_fn: Optional[Callable[..., Any]] = None,
    build_pool_fn: Optional[Callable[..., Pool]] = None,
) -> dict[str, Any]:
    import pandas as pd

    from ..data.split import subject_independent_split
    from ..metrics.learning_curve import area_under_learning_curve

    strategy_kwargs = dict(strategy_kwargs or {})
    build_pool = build_pool_fn or (
        lambda df, preds, emb: _build_pool(
            df, preds, emb, sample_id_col=sample_id_col
        )
    )
    sorted_budgets = sorted(float(b) for b in budgets)
    B = len(sorted_budgets)
    seeds = [int(s) for s in seeds]
    strategy_name = (
        strategy if isinstance(strategy, str) else getattr(strategy, "name", "custom")
    )

    scores = np.full((len(seeds), B), np.nan, dtype=np.float64)
    n_labeled_grid = np.zeros((len(seeds), B), dtype=np.int64)
    subject_splits: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []

    for si, seed in enumerate(seeds):
        split_df = subject_independent_split(
            manifest_df, ratios=split_ratios, seed=seed, group_col=group_col
        )
        pool_df = split_df[split_df["split"].isin(list(pool_splits))].reset_index(
            drop=True
        )
        test_df = split_df[split_df["split"] == test_split].reset_index(drop=True)

        pool_ids = [str(s) for s in pool_df[sample_id_col].tolist()]
        N = len(pool_ids)
        hidden = {sid: sid for sid in pool_ids}

        def _subjects(d: Any) -> list[Any]:
            if group_col in d.columns:
                return sorted({str(x) for x in d[group_col].tolist()})
            return []

        subject_splits.append(
            {
                "seed": seed,
                "n_pool": N,
                "n_test": len(test_df),
                "pool_subjects": _subjects(pool_df),
                "test_subjects": _subjects(test_df),
            }
        )

        strat = _resolve_strategy(strategy, seed, strategy_kwargs)
        init_strat = _resolve_strategy(init_strategy, seed, {})

        labeled: set[str] = set()
        model: Any = None

        def _grow_to(target_n: int) -> None:
            nonlocal labeled
            target_n = max(0, min(int(target_n), N))
            guard = 0
            while len(labeled) < target_n and guard <= N:
                guard += 1
                preds = None
                emb = None
                if model is not None:
                    preds = _predictions_for_df(
                        infer_fn(model, pool_df),
                        pool_ids,
                        num_keypoints=num_keypoints,
                    )
                    if embed_fn is not None:
                        emb = np.asarray(embed_fn(model, pool_df), dtype=np.float64)
                pool = build_pool(pool_df, preds, emb)
                k = target_n - len(labeled)
                try:
                    res = run_al_loop_step(
                        pool, strat, k, labeled_ids=labeled,
                        hidden_labels=hidden, mode="retrospective",
                    )
                except ValueError:
                    res = run_al_loop_step(
                        pool, init_strat, k, labeled_ids=labeled,
                        hidden_labels=hidden, mode="retrospective",
                    )
                if len(res.labeled_ids) <= len(labeled):
                    break
                labeled = res.labeled_ids

        if initial_fraction > 0:
            _grow_to(round(initial_fraction * N))

        seed_budgets: list[float] = []
        seed_scores: list[float] = []
        seed_nlabeled: list[int] = []
        for bi, budget in enumerate(sorted_budgets):
            target_n = N if budget >= 1.0 else max(1, int(round(budget * N)))
            _grow_to(target_n)

            labeled_df = pool_df[
                pool_df[sample_id_col].astype(str).isin(labeled)
            ].reset_index(drop=True)
            model = train_fn(
                labeled_df,
                labeled_ids=set(labeled),
                seed=seed,
                budget=budget,
                round_idx=bi,
            )
            test_pred = _predictions_for_df(
                infer_fn(model, test_df),
                [str(s) for s in test_df[sample_id_col].tolist()]
                if sample_id_col in test_df.columns
                else list(range(len(test_df))),
                num_keypoints=num_keypoints,
            )
            metric = evaluate_fn(
                model,
                test_df,
                test_pred,
                seed=seed,
                budget=budget,
                n_labeled=len(labeled),
                pool_size=N,
            )
            val = _metric_scalar(metric, metric_name)
            scores[si, bi] = val
            n_labeled_grid[si, bi] = len(labeled)
            seed_budgets.append(budget)
            seed_scores.append(val)
            seed_nlabeled.append(len(labeled))

        per_seed.append(
            {
                "seed": seed,
                "budgets": seed_budgets,
                "scores": seed_scores,
                "n_labeled": seed_nlabeled,
            }
        )

    mean = np.nanmean(scores, axis=0)
    std = np.nanstd(scores, axis=0)

    auc_per_seed = [
        area_under_learning_curve(sorted_budgets, scores[si], normalize=False)
        for si in range(len(seeds))
    ]
    auc = area_under_learning_curve(sorted_budgets, mean, normalize=False)
    auc_norm = area_under_learning_curve(sorted_budgets, mean, normalize=True)

    points = [
        {
            "budget": sorted_budgets[bi],
            "mean": float(mean[bi]),
            "std": float(std[bi]),
            "scores": [float(scores[si, bi]) for si in range(len(seeds))],
            "n_labeled": [int(n_labeled_grid[si, bi]) for si in range(len(seeds))],
        }
        for bi in range(B)
    ]

    monotone = bool(np.all(np.diff(mean) >= -1e-9)) if B > 1 else True

    return {
        "strategy": strategy_name,
        "init_strategy": init_strategy,
        "budgets": sorted_budgets,
        "seeds": seeds,
        "metric_name": metric_name,
        "scores": scores,
        "n_labeled": n_labeled_grid,
        "mean": mean,
        "std": std,
        "points": points,
        "auc": float(auc) if np.isfinite(auc) else float("nan"),
        "auc_normalized": float(auc_norm) if np.isfinite(auc_norm) else float("nan"),
        "auc_per_seed": [float(a) for a in auc_per_seed],
        "subject_splits": subject_splits,
        "per_seed": per_seed,
        "monotone_increasing": monotone,
    }
