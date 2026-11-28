from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import AcquisitionStrategy, Pool

__all__ = ["RandomStrategy"]


class RandomStrategy(AcquisitionStrategy):
    name = "random"

    def score(self, pool: Pool) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random(pool.size)

    def select(
        self,
        pool: Pool,
        k: int,
        *,
        exclude: Optional[Sequence[int]] = None,
        candidates: Optional[Sequence[int]] = None,
    ) -> list[int]:
        n = pool.size
        cand = list(range(n)) if candidates is None else [int(i) for i in candidates]
        if exclude:
            ex = set(int(i) for i in exclude)
            cand = [i for i in cand if i not in ex]
        if k <= 0 or not cand:
            return []
        k = min(k, len(cand))
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(len(cand), size=k, replace=False)
        return [cand[int(i)] for i in chosen]
