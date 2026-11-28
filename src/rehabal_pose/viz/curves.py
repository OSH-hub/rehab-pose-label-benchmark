from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as _plt  # type: ignore

    HAS_MPL = True
except Exception:  # pragma: no cover
    _plt = None  # type: ignore
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise RuntimeError(
            "matplotlib is required for curve plotting; install it with "
            "`pip install matplotlib`."
        )


def plot_budget_curve(
    budgets: Sequence[float],
    series: Mapping[str, np.ndarray | Sequence[float] | Sequence[Sequence[float]]],
    *,
    ax: Optional[Any] = None,
    ylabel: str = "accuracy",
    xlabel: str = "label budget (fraction)",
    title: Optional[str] = None,
) -> Any:
    _require_mpl()
    if ax is None:
        _, ax = _plt.subplots()
    x = np.asarray(budgets, dtype=float)
    for name, scores in series.items():
        arr = np.asarray(scores, dtype=float)
        if arr.ndim == 1:
            ax.plot(x, arr, marker="o", label=name)
        else:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(x, mean, marker="o", label=name)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax
