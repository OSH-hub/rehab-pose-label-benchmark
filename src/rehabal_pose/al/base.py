from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "Pool",
    "AcquisitionStrategy",
    "minmax_normalize",
]


def minmax_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return a
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    rng = hi - lo
    if rng < eps:
        return np.zeros_like(a)
    out = (a - lo) / rng
    return np.clip(out, 0.0, 1.0)


@dataclass
class Pool:
    sample_ids: Sequence[str]
    predictions: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    clip_ids: Optional[Sequence[Any]] = None
    frame_ids: Optional[Sequence[int]] = None
    metadata: Optional[dict[str, Sequence[Any]]] = None
    prediction_samples: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.sample_ids = list(self.sample_ids)
        n = len(self.sample_ids)

        if self.predictions is not None:
            self.predictions = np.asarray(self.predictions, dtype=np.float64)
            if self.predictions.ndim != 3 or self.predictions.shape[2] != 3:
                raise ValueError(
                    "Pool.predictions must have shape (N, K, 3) [x, y, conf]; "
                    f"got {self.predictions.shape}"
                )
            if self.predictions.shape[0] != n:
                raise ValueError(
                    f"Pool.predictions has {self.predictions.shape[0]} rows but "
                    f"there are {n} sample_ids"
                )

        if self.embeddings is not None:
            self.embeddings = np.asarray(self.embeddings, dtype=np.float64)
            if self.embeddings.ndim != 2:
                raise ValueError(
                    f"Pool.embeddings must be 2-D (N, D); got {self.embeddings.shape}"
                )
            if self.embeddings.shape[0] != n:
                raise ValueError(
                    f"Pool.embeddings has {self.embeddings.shape[0]} rows but "
                    f"there are {n} sample_ids"
                )

        if self.prediction_samples is not None:
            self.prediction_samples = np.asarray(
                self.prediction_samples, dtype=np.float64
            )
            if self.prediction_samples.ndim != 4 or self.prediction_samples.shape[
                3
            ] != 3:
                raise ValueError(
                    "Pool.prediction_samples must have shape (N, T, K, 3); "
                    f"got {self.prediction_samples.shape}"
                )
            if self.prediction_samples.shape[0] != n:
                raise ValueError(
                    "Pool.prediction_samples first axis must equal N"
                )

        for attr in ("clip_ids", "frame_ids"):
            seq = getattr(self, attr)
            if seq is not None:
                seq = list(seq)
                if len(seq) != n:
                    raise ValueError(f"Pool.{attr} length {len(seq)} != N {n}")
                setattr(self, attr, seq)

        if self.metadata is not None:
            for key, col in self.metadata.items():
                if len(list(col)) != n:
                    raise ValueError(
                        f"Pool.metadata[{key!r}] length {len(list(col))} != N {n}"
                    )


    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def size(self) -> int:
        return len(self.sample_ids)


    def keypoints_xy(self) -> np.ndarray:
        if self.predictions is None:
            raise ValueError("Pool has no predictions; cannot extract keypoints_xy")
        return self.predictions[..., :2]

    def confidence(self) -> np.ndarray:
        if self.predictions is None:
            raise ValueError("Pool has no predictions; cannot extract confidence")
        return self.predictions[..., 2]

    def mean_confidence(self) -> np.ndarray:
        return self.confidence().mean(axis=1)


    def meta(self, key: str) -> Optional[list[Any]]:
        if self.metadata is None or key not in self.metadata:
            return None
        return list(self.metadata[key])


    def clip_groups(self) -> dict[Any, list[int]]:
        n = self.size
        if self.clip_ids is None:
            return {self.sample_ids[i]: [i] for i in range(n)}

        groups: dict[Any, list[int]] = {}
        for i in range(n):
            groups.setdefault(self.clip_ids[i], []).append(i)

        if self.frame_ids is not None:
            for cid, idxs in groups.items():
                idxs.sort(key=lambda i: (self.frame_ids[i], i))
        return groups


class AcquisitionStrategy(ABC):
    name: str = "abstract"

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        diversify: Optional[str] = None,
        diversity_lambda: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if diversify not in (None, "mmr", "kcenter"):
            raise ValueError(
                f"diversify must be None, 'mmr', or 'kcenter'; got {diversify!r}"
            )
        self.seed = seed
        self.diversify = diversify
        self.diversity_lambda = float(diversity_lambda)
        self.config: dict[str, Any] = dict(kwargs)


    @abstractmethod
    def score(self, pool: Pool) -> np.ndarray:
        raise NotImplementedError


    def select(
        self,
        pool: Pool,
        k: int,
        *,
        exclude: Optional[Sequence[int]] = None,
        candidates: Optional[Sequence[int]] = None,
    ) -> list[int]:
        n = pool.size
        if candidates is None:
            cand = list(range(n))
        else:
            cand = [int(i) for i in candidates]
        if exclude:
            ex = set(int(i) for i in exclude)
            cand = [i for i in cand if i not in ex]
        if k <= 0 or not cand:
            return []
        k = min(k, len(cand))

        scores = np.asarray(self.score(pool), dtype=np.float64)
        cand_scores = scores[cand]

        if self.diversify is not None and pool.embeddings is not None:
            from .diversity import kcenter_select, mmr_select

            emb = pool.embeddings[cand]
            if self.diversify == "mmr":
                local = mmr_select(
                    emb, cand_scores, k, lambda_=self.diversity_lambda
                )
            else:
                local = kcenter_select(emb, k, seed=self.seed)
            return [cand[i] for i in local]

        order = np.lexsort((np.arange(len(cand)), -cand_scores))
        chosen_local = order[:k]
        return [cand[i] for i in chosen_local]
