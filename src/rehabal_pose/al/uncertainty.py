"""Uncertainty-based acquisition.

Implements least-confidence sampling over keypoint detection confidence, with
optional predictive-variance from MC-dropout / test-time-augmentation (TTA)
samples when available on the pool.
"""

from __future__ import annotations

import numpy as np

from .base import AcquisitionStrategy, Pool, minmax_normalize

__all__ = ["UncertaintyStrategy"]


class UncertaintyStrategy(AcquisitionStrategy):
    """Least-confidence (optionally variance-augmented) uncertainty sampling.

    The base uncertainty of a sample is ``1 - mean_keypoint_confidence``, i.e.
    low-confidence predictions score high.

    If ``Pool.prediction_samples`` (shape ``(N, T, K, 3)`` from MC-dropout or
    TTA) is present, the per-keypoint positional variance across the ``T``
    stochastic passes is min-max normalized and blended in with weight
    ``variance_weight``.

    Parameters
    ----------
    variance_weight:
        Blend weight in ``[0, 1]`` for the MC/TTA variance term. Ignored when
        no ``prediction_samples`` are available.
    """

    name = "uncertainty"

    def __init__(self, *, variance_weight: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.variance_weight = float(variance_weight)

    def score(self, pool: Pool) -> np.ndarray:
        """Per-sample uncertainty score in roughly ``[0, 1]`` (higher = less
        confident / higher variance)."""
        if pool.predictions is None:
            raise ValueError("UncertaintyStrategy requires pool.predictions")

        # Least confidence: 1 - mean confidence over keypoints.
        least_conf = 1.0 - pool.mean_confidence()
        score = least_conf.astype(np.float64)

        # Optional MC-dropout / TTA predictive variance over (x, y).
        if pool.prediction_samples is not None and self.variance_weight > 0:
            xy = pool.prediction_samples[..., :2]  # (N, T, K, 2)
            var = xy.var(axis=1)  # (N, K, 2)
            per_sample_var = var.mean(axis=(1, 2))  # (N,)
            var_n = minmax_normalize(per_sample_var)
            w = self.variance_weight
            score = (1.0 - w) * score + w * var_n

        return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
