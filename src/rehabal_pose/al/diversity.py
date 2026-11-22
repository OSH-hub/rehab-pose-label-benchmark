"""Diversity / de-duplication helpers for batch active learning.

Two building blocks used by the strategies' ``select`` step:

* :func:`kcenter_select` — greedy k-center (farthest-first) selection that
  spreads chosen points across the embedding space (the core-set heuristic).
* :func:`mmr_select` — Maximal Marginal Relevance, trading off per-sample
  relevance (e.g. an uncertainty score) against redundancy with already
  chosen samples.

Both operate purely on NumPy arrays and are deterministic given their inputs
(``kcenter_select`` seeds its first pick from the point farthest from the
centroid, falling back to the RNG only for degenerate inputs).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["pairwise_distances", "kcenter_select", "mmr_select"]


def pairwise_distances(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """Euclidean distance matrix between rows of *a* and *b*.

    Parameters
    ----------
    a:
        ``(M, D)`` array.
    b:
        ``(N, D)`` array; defaults to *a* (yielding an ``(M, M)`` matrix).

    Returns
    -------
    numpy.ndarray
        ``(M, N)`` non-negative distance matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    if b is None:
        b = a
    else:
        b = np.asarray(b, dtype=np.float64)
    # Memory-efficient ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b^T -> (M, N) only,
    # never materializing the (M, N, D) difference (which blows up for large N:
    # e.g. 8367x8367x256 float64 = 134 GiB).
    a2 = np.einsum("md,md->m", a, a)
    b2 = np.einsum("nd,nd->n", b, b)
    d2 = a2[:, None] + b2[None, :] - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def kcenter_select(
    embeddings: np.ndarray,
    k: int,
    *,
    initial_indices: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> list[int]:
    """Greedy k-center (farthest-first traversal) on *embeddings*.

    Iteratively picks the point whose minimum distance to the already-selected
    set is largest, which spreads selections to cover the feature space — the
    classic core-set acquisition heuristic.

    Parameters
    ----------
    embeddings:
        ``(N, D)`` feature matrix.
    k:
        Number of points to select (clamped to ``N``).
    initial_indices:
        Optional indices of an existing labeled set. New picks maximize
        coverage relative to these *and* to subsequently chosen points; the
        returned list contains only the newly selected indices.
    seed:
        RNG seed used only to break the (rare) all-equal-distance ties.

    Returns
    -------
    list[int]
        Selected indices into *embeddings* (length ``min(k, N)``).
    """
    X = np.asarray(embeddings, dtype=np.float64)
    n = X.shape[0]
    k = min(int(k), n)
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)

    selected: list[int] = []
    available = np.ones(n, dtype=bool)

    if initial_indices:
        init = [int(i) for i in initial_indices if 0 <= int(i) < n]
        if init:
            min_dist = pairwise_distances(X, X[init]).min(axis=1)
        else:
            min_dist = np.full(n, np.inf)
    else:
        min_dist = np.full(n, np.inf)

    # First pick: farthest from centroid (deterministic) when no seed set.
    if not (initial_indices and len(list(initial_indices)) > 0):
        centroid = X.mean(axis=0, keepdims=True)
        d0 = pairwise_distances(X, centroid).ravel()
        if np.ptp(d0) < 1e-12:
            first = int(rng.integers(n))
        else:
            first = int(np.argmax(d0))
        selected.append(first)
        available[first] = False
        min_dist = np.minimum(min_dist, pairwise_distances(X, X[[first]]).ravel())

    while len(selected) < k:
        masked = np.where(available, min_dist, -np.inf)
        if not np.any(np.isfinite(masked) & available):
            # All remaining equidistant/degenerate: pick any available.
            remaining = np.flatnonzero(available)
            nxt = int(rng.choice(remaining))
        else:
            nxt = int(np.argmax(masked))
        selected.append(nxt)
        available[nxt] = False
        min_dist = np.minimum(min_dist, pairwise_distances(X, X[[nxt]]).ravel())

    return selected


def mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    k: int,
    *,
    lambda_: float = 0.5,
) -> list[int]:
    """Maximal Marginal Relevance selection.

    Greedily picks the sample maximizing
    ``lambda_ * relevance - (1 - lambda_) * max_similarity_to_selected``,
    balancing informativeness against redundancy. Similarity is derived from
    Euclidean distance scaled into ``[0, 1]`` (``sim = 1 - d / d_max``).

    Parameters
    ----------
    embeddings:
        ``(N, D)`` feature matrix.
    relevance:
        ``(N,)`` per-sample relevance/uncertainty scores (any scale; they are
        min-max normalized internally).
    k:
        Number of points to select (clamped to ``N``).
    lambda_:
        Trade-off in ``[0, 1]`` (1.0 = pure relevance, 0.0 = pure diversity).

    Returns
    -------
    list[int]
        Selected indices (length ``min(k, N)``).
    """
    X = np.asarray(embeddings, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64).ravel()
    n = X.shape[0]
    k = min(int(k), n)
    if k <= 0:
        return []
    lam = float(np.clip(lambda_, 0.0, 1.0))

    # Normalize relevance to [0, 1].
    rng = np.ptp(rel)
    rel_n = (rel - rel.min()) / rng if rng > 1e-12 else np.zeros_like(rel)

    # Similarity in [0, 1] from distances.
    D = pairwise_distances(X)
    dmax = float(D.max())
    sim = 1.0 - D / dmax if dmax > 1e-12 else np.ones_like(D)

    selected: list[int] = [int(np.argmax(rel_n))]
    available = np.ones(n, dtype=bool)
    available[selected[0]] = False

    while len(selected) < k:
        max_sim = sim[:, selected].max(axis=1)
        mmr = lam * rel_n - (1.0 - lam) * max_sim
        mmr[~available] = -np.inf
        nxt = int(np.argmax(mmr))
        selected.append(nxt)
        available[nxt] = False

    return selected
