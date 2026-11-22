"""Active-learning core: the :class:`Pool` container and the
:class:`AcquisitionStrategy` abstract base class.

A :class:`Pool` carries everything an acquisition function might need to score
unlabeled samples:

* ``sample_ids``    — stable identifiers (length ``N``).
* ``predictions``   — model output keypoints + confidence, shape ``(N, K, 3)``
  as ``[x, y, confidence]``.
* ``embeddings``    — optional feature vectors, shape ``(N, D)``.
* ``clip_ids`` / ``frame_ids`` — optional temporal grouping/ordering.
* ``metadata``      — optional column-oriented dict of per-sample metadata
  (e.g. ``{"disease": [...], "exercise": [...], "camera_id": [...]}``).
* ``prediction_samples`` — optional MC-dropout / TTA stochastic predictions,
  shape ``(N, T, K, 3)``, used to estimate predictive variance.

Only NumPy is imported here; the module is import-safe for the test suite (no
torch / cv2 / ultralytics).
"""

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


# ---------------------------------------------------------------------------
# Small numeric helper
# ---------------------------------------------------------------------------


def minmax_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Min-max scale a 1-D array into ``[0, 1]``.

    If the value range is (near) zero the function returns all-zeros so the
    term contributes no signal (rather than producing ``NaN``).

    Parameters
    ----------
    x:
        1-D array of finite values.
    eps:
        Range threshold below which the input is treated as constant.

    Returns
    -------
    numpy.ndarray
        ``float64`` array in ``[0, 1]`` of the same shape as *x*.
    """
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


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


@dataclass
class Pool:
    """A pool of (unlabeled) candidate samples for active-learning queries.

    All array-like attributes are aligned along the first axis of length
    ``N == len(sample_ids)``. Optional attributes may be ``None`` when the
    corresponding information is unavailable; individual strategies document
    which attributes they require.
    """

    sample_ids: Sequence[str]
    predictions: Optional[np.ndarray] = None  # (N, K, 3) [x, y, confidence]
    embeddings: Optional[np.ndarray] = None  # (N, D)
    clip_ids: Optional[Sequence[Any]] = None  # (N,)
    frame_ids: Optional[Sequence[int]] = None  # (N,)
    metadata: Optional[dict[str, Sequence[Any]]] = None  # column-oriented
    prediction_samples: Optional[np.ndarray] = None  # (N, T, K, 3) MC/TTA

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

    # -- size --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sample_ids)

    @property
    def size(self) -> int:
        """Number of samples ``N`` in the pool."""
        return len(self.sample_ids)

    # -- prediction-derived views -----------------------------------------

    def keypoints_xy(self) -> np.ndarray:
        """Return keypoint coordinates ``(N, K, 2)`` from ``predictions``.

        Raises
        ------
        ValueError
            If ``predictions`` is not set.
        """
        if self.predictions is None:
            raise ValueError("Pool has no predictions; cannot extract keypoints_xy")
        return self.predictions[..., :2]

    def confidence(self) -> np.ndarray:
        """Per-keypoint confidence channel ``(N, K)`` from ``predictions``."""
        if self.predictions is None:
            raise ValueError("Pool has no predictions; cannot extract confidence")
        return self.predictions[..., 2]

    def mean_confidence(self) -> np.ndarray:
        """Mean per-sample confidence ``(N,)`` over keypoints."""
        return self.confidence().mean(axis=1)

    # -- metadata ----------------------------------------------------------

    def meta(self, key: str) -> Optional[list[Any]]:
        """Return the metadata column *key* as a list, or ``None`` if absent."""
        if self.metadata is None or key not in self.metadata:
            return None
        return list(self.metadata[key])

    # -- temporal grouping -------------------------------------------------

    def clip_groups(self) -> dict[Any, list[int]]:
        """Group sample indices by ``clip_id``, ordered by ``frame_id``.

        When ``clip_ids`` is ``None`` every sample is treated as its own
        singleton clip (keyed by ``sample_id``). When ``frame_ids`` is ``None``
        the natural pool order is preserved within each clip.

        Returns
        -------
        dict
            Mapping ``clip_id -> [indices ordered by frame]``.
        """
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


# ---------------------------------------------------------------------------
# AcquisitionStrategy
# ---------------------------------------------------------------------------


class AcquisitionStrategy(ABC):
    """Abstract base class for active-learning acquisition strategies.

    Concrete strategies implement :meth:`score`, returning a per-sample
    informativeness score (higher = more worth labeling). The base
    :meth:`select` implements *top-k* selection with an optional
    diversity/de-duplication hook (MMR or k-center on embeddings), so the
    common case requires implementing only :meth:`score`.
    """

    #: Registry key / human-readable strategy name; subclasses override.
    name: str = "abstract"

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        diversify: Optional[str] = None,
        diversity_lambda: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Parameters
        ----------
        seed:
            Optional RNG seed for any stochastic behavior.
        diversify:
            Diversity hook used by :meth:`select`: ``None`` (pure top-k),
            ``"mmr"`` (Maximal Marginal Relevance), or ``"kcenter"``
            (k-center greedy). Requires ``Pool.embeddings`` to take effect.
        diversity_lambda:
            MMR trade-off in ``[0, 1]`` (1.0 = pure relevance, 0.0 = pure
            diversity).
        """
        if diversify not in (None, "mmr", "kcenter"):
            raise ValueError(
                f"diversify must be None, 'mmr', or 'kcenter'; got {diversify!r}"
            )
        self.seed = seed
        self.diversify = diversify
        self.diversity_lambda = float(diversity_lambda)
        self.config: dict[str, Any] = dict(kwargs)

    # -- scoring -----------------------------------------------------------

    @abstractmethod
    def score(self, pool: Pool) -> np.ndarray:
        """Return a ``(N,)`` float array of per-sample informativeness scores.

        Higher means *more* worth labeling. Implementations must return finite
        values for every sample.
        """
        raise NotImplementedError

    # -- selection ---------------------------------------------------------

    def select(
        self,
        pool: Pool,
        k: int,
        *,
        exclude: Optional[Sequence[int]] = None,
        candidates: Optional[Sequence[int]] = None,
    ) -> list[int]:
        """Select up to *k* sample indices to query.

        Default behavior is top-*k* by :meth:`score`, optionally passed through
        an MMR or k-center diversity filter (the de-dup hook) when ``diversify``
        is configured and ``Pool.embeddings`` is available.

        Parameters
        ----------
        pool:
            The candidate pool.
        k:
            Query budget (clamped to the number of available candidates).
        exclude:
            Indices to exclude from selection (e.g. already-labeled samples).
        candidates:
            If given, restrict selection to these indices.

        Returns
        -------
        list[int]
            Unique selected indices into *pool* (length ``min(k, available)``).
        """
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

        # Diversity-aware selection on embeddings (the MMR / k-center hook).
        if self.diversify is not None and pool.embeddings is not None:
            from .diversity import kcenter_select, mmr_select

            emb = pool.embeddings[cand]
            if self.diversify == "mmr":
                local = mmr_select(
                    emb, cand_scores, k, lambda_=self.diversity_lambda
                )
            else:  # kcenter
                local = kcenter_select(emb, k, seed=self.seed)
            return [cand[i] for i in local]

        # Plain top-k (stable: ties broken by ascending index).
        order = np.lexsort((np.arange(len(cand)), -cand_scores))
        chosen_local = order[:k]
        return [cand[i] for i in chosen_local]
