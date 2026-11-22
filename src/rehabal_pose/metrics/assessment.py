"""Exercise-quality / clinical-assessment metrics (pure NumPy).

These compare model-estimated rehabilitation-quality outputs against clinician
labels:

* :func:`quality_score_mae` -- MAE between continuous quality scores.
* :func:`quality_score_corr` -- Pearson or Spearman correlation of scores.
* :func:`compensation_f1` -- F1 for detecting compensatory movements
  (binary or multilabel).
* :func:`proxy_deviation_score` -- an **objective PROXY** for movement deviation
  from a reference sequence, usable when no clinician labels are available.

The functions compute exact math when inputs are supplied. They are intended to
be wired into the evaluation pipeline; where ground-truth clinician labels are
unavailable the proxy is used as a stand-in (clearly marked).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "quality_score_mae",
    "quality_score_corr",
    "compensation_f1",
    "proxy_deviation_score",
]

_EPS = 1e-12


def quality_score_mae(pred_scores: Any, gt_scores: Any) -> float:
    """Mean absolute error between predicted and clinician quality scores.

    Parameters
    ----------
    pred_scores, gt_scores:
        1D sequences of equal length.

    Returns
    -------
    float
        ``mean(|pred - gt|)``. NaN if there are no samples.
    """
    p = np.asarray(pred_scores, dtype=np.float64).ravel()
    g = np.asarray(gt_scores, dtype=np.float64).ravel()
    if p.shape != g.shape:
        raise ValueError(f"score length mismatch: {p.shape} vs {g.shape}")
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - g)))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2))
    if denom < _EPS:
        return float("nan")
    return float(np.sum(xc * yc) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), 1-based, NumPy-only (SciPy-free)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    sorted_a = a[order]
    i = 0
    n = a.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def quality_score_corr(
    pred_scores: Any, gt_scores: Any, method: str = "pearson"
) -> float:
    """Correlation between predicted and clinician quality scores.

    Parameters
    ----------
    pred_scores, gt_scores:
        1D sequences of equal length (>= 2 elements).
    method:
        ``"pearson"`` (linear) or ``"spearman"`` (rank). Case-insensitive.

    Returns
    -------
    float
        Correlation coefficient in ``[-1, 1]``; NaN if undefined (constant
        input or < 2 samples).
    """
    p = np.asarray(pred_scores, dtype=np.float64).ravel()
    g = np.asarray(gt_scores, dtype=np.float64).ravel()
    if p.shape != g.shape:
        raise ValueError(f"score length mismatch: {p.shape} vs {g.shape}")
    if p.size < 2:
        return float("nan")

    m = method.lower()
    if m == "pearson":
        return _pearson(p, g)
    if m == "spearman":
        return _pearson(_rankdata(p), _rankdata(g))
    raise ValueError(f"Unknown correlation method: {method!r}")


def compensation_f1(
    pred_labels: Any,
    gt_labels: Any,
    *,
    average: str = "binary",
    positive_label: Any = 1,
) -> dict[str, float]:
    """F1 score for compensatory-movement detection.

    Supports binary labels (``average="binary"``, default), and multilabel /
    multiclass one-hot or integer arrays via ``average in {"micro", "macro"}``.

    Parameters
    ----------
    pred_labels, gt_labels:
        Equal-shaped label arrays. For ``"binary"`` they are 1D and the
        positive class is ``positive_label``. For ``"micro"``/``"macro"`` they
        are 2D ``(N, C)`` binary indicator arrays.
    average:
        ``"binary"``, ``"micro"`` or ``"macro"``.
    positive_label:
        Value treated as positive in binary mode.

    Returns
    -------
    dict
        ``{"precision": float, "recall": float, "f1": float}``.
    """
    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    if pred.shape != gt.shape:
        raise ValueError(f"label shape mismatch: {pred.shape} vs {gt.shape}")

    def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    avg = average.lower()
    if avg == "binary":
        p = pred.ravel() == positive_label
        g = gt.ravel() == positive_label
        tp = float(np.sum(p & g))
        fp = float(np.sum(p & ~g))
        fn = float(np.sum(~p & g))
        prec, rec, f1 = _prf(tp, fp, fn)
        return {"precision": prec, "recall": rec, "f1": f1}

    if pred.ndim != 2:
        raise ValueError(f"{avg} averaging requires 2D (N, C) indicator arrays")
    pb = pred.astype(bool)
    gb = gt.astype(bool)
    if avg == "micro":
        tp = float(np.sum(pb & gb))
        fp = float(np.sum(pb & ~gb))
        fn = float(np.sum(~pb & gb))
        prec, rec, f1 = _prf(tp, fp, fn)
        return {"precision": prec, "recall": rec, "f1": f1}
    if avg == "macro":
        precs, recs, f1s = [], [], []
        for c in range(pb.shape[1]):
            tp = float(np.sum(pb[:, c] & gb[:, c]))
            fp = float(np.sum(pb[:, c] & ~gb[:, c]))
            fn = float(np.sum(~pb[:, c] & gb[:, c]))
            pr, re, f = _prf(tp, fp, fn)
            precs.append(pr)
            recs.append(re)
            f1s.append(f)
        return {
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
        }
    raise ValueError(f"Unknown average: {average!r}")


def _resample_to(seq: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample ``seq`` (shape ``(T, ...)``) to ``length`` frames."""
    t = seq.shape[0]
    if t == length:
        return seq
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, length)
    flat = seq.reshape(t, -1)
    out = np.empty((length, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape((length,) + seq.shape[1:])


def proxy_deviation_score(seq: Any, reference_seq: Any) -> float:
    """PROXY objective deviation of a movement from a reference sequence.

    .. warning::
       **PROXY METRIC.** This is *not* a clinically validated quality score.
       It is an objective, label-free stand-in: the mean per-frame, per-joint
       Euclidean distance between ``seq`` and ``reference_seq`` after linear
       length alignment. Use only as a relative/automated signal (e.g. for
       active-learning acquisition or sanity checks), never as a clinical
       outcome.

    Parameters
    ----------
    seq, reference_seq:
        Sequences shaped ``(T, K, D)`` (or ``(T, F)``). They may differ in
        length; the shorter is resampled to the longer via linear interpolation.

    Returns
    -------
    float
        Non-negative mean deviation (same units as the coordinates). ``0.0``
        means identical motion; larger means more deviation.
    """
    s = np.asarray(seq, dtype=np.float64)
    r = np.asarray(reference_seq, dtype=np.float64)
    if s.ndim == 2:
        s = s[:, :, None]
    if r.ndim == 2:
        r = r[:, :, None]
    if s.shape[1:] != r.shape[1:]:
        raise ValueError(
            f"sequence feature shapes differ: {s.shape[1:]} vs {r.shape[1:]}"
        )
    length = max(s.shape[0], r.shape[0])
    if length == 0:
        return float("nan")
    s2 = _resample_to(s, length)
    r2 = _resample_to(r, length)
    per_frame_joint = np.linalg.norm(s2 - r2, axis=-1)  # (length, K)
    return float(np.mean(per_frame_joint))
