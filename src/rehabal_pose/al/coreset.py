from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import AcquisitionStrategy, Pool
from .diversity import kcenter_select, pairwise_distances

__all__ = ["CoresetStrategy"]


class CoresetStrategy(AcquisitionStrategy):
    name = "coreset"

    def __init__(self, *, labeled_embeddings: Optional[np.ndarray] = None, **kwargs):
        super().__init__(**kwargs)
        self.labeled_embeddings = (
            np.asarray(labeled_embeddings, dtype=np.float64)
            if labeled_embeddings is not None
            else None
        )

    def score(self, pool: Pool) -> np.ndarray:
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

        initial = None
        if self.labeled_embeddings is not None and len(self.labeled_embeddings):
            stacked = np.vstack([emb, self.labeled_embeddings])
            initial = list(range(len(cand), len(cand) + len(self.labeled_embeddings)))
            local = kcenter_select(stacked, k, initial_indices=initial, seed=self.seed)
            local = [i for i in local if i < len(cand)][:k]
        else:
            local = kcenter_select(emb, k, seed=self.seed)

        return [cand[i] for i in local]
