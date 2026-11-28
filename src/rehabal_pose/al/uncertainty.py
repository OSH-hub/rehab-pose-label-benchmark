from __future__ import annotations

import numpy as np

from .base import AcquisitionStrategy, Pool, minmax_normalize

__all__ = ["UncertaintyStrategy"]


class UncertaintyStrategy(AcquisitionStrategy):
    name = "uncertainty"

    def __init__(self, *, variance_weight: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variance_weight = float(variance_weight)

    def score(self, pool: Pool) -> np.ndarray:
        if pool.predictions is None:
            raise ValueError("UncertaintyStrategy requires pool.predictions")

        least_conf = 1.0 - pool.mean_confidence()
        score = least_conf.astype(np.float64)

        if pool.prediction_samples is not None and self.variance_weight > 0:
            xy = pool.prediction_samples[..., :2]
            var = xy.var(axis=1)
            per_sample_var = var.mean(axis=(1, 2))
            var_n = minmax_normalize(per_sample_var)
            w = self.variance_weight
            score = (1.0 - w) * score + w * var_n

        return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
