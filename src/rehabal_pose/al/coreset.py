"""Core-set acquisition (k-center greedy on embeddings).

Selects a representative, well-spread subset of the pool in feature space,
following the core-set view of active learning (Sener & Savarese, 2018).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import AcquisitionStrategy, Pool
from .diversity import kcenter_select, pairwise_distances

__all__ = ["CoresetStrategy"]


class CoresetStrategy(AcquisitionStrategy):
    """k-center greedy core-set selection over ``Pool.embeddings``.

    The set-level objective (cover the feature space) is realized in
    :meth:`select`. :meth:`score` exposes a per-sample proxy — distance to the
    embedding centroid — so the strategy still satisfies the scoring contract
    and degrades gracefully to top-k when used without its own ``select``.

    Parameters
    ----------
    labeled_embeddings:
        Optional ``(M, D)`` embeddings of an already-labeled set; new picks
        maximize coverage relative to them.
    """

    name = "coreset"

    def __init__(self, *, labeled_embeddings: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        self.labeled_embeddings = (
            np.asarray(labeled_embeddings, dtype=np.float64)
            if labeled_embeddings is not None
            else None
        )

    def score(self, pool: Pool) -> np.ndarray:
        """Per-sample distance to the embedding centroid (representativeness
        proxy; larger = more peripheral/novel)."""
        if pool.embeddings is None:
            raise ValueError("CoresetStrategy requires pool.embeddings")
        centroid = pool.embeddings.mean(axis=0, keepdims=True)
        return pairwise_distances(pool.embeddings, centroid).ravel()

    def select(
        self,
        pool: Pool,
        k: int,
        *,
        exclude: Optional[Sequence[int]] = None,
        candidates: Optional[Sequence[int]] = None,
    ) -> list[int]:
        """Greedy k-center selection over candidate embeddings."""
        if pool.embeddings is None:
            raise ValueError("CoresetStrategy requires pool.embeddings")

        n = pool.size
        cand = list(range(n)) if candidates is None else [int(i) for i in candidates]
        if exclude:
            ex = set(int(i) for i in exclude)
            cand = [i for i in cand if i not in ex]
        if k <= 0 or not cand:
            return []
        k = min(k, len(cand))

        emb = pool.embeddings[cand]

        # Seed coverage with already-labeled points if provided.
        initial = None
        if self.labeled_embeddings is not None and len(self.labeled_embeddings):
            stacked = np.vstack([emb, self.labeled_embeddings])
            initial = list(range(len(cand), len(cand) + len(self.labeled_embeddings)))
            local = kcenter_select(stacked, k, initial_indices=initial, seed=self.seed)
            # Drop any (defensively) returned labeled-region indices.
            local = [i for i in local if i < len(cand)][:k]
        else:
            local = kcenter_select(emb, k, seed=self.seed)

        return [cand[i] for i in local]
