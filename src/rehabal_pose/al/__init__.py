"""Active-learning subpackage for RehabAL-Pose.

Exposes the :class:`Pool` container, the :class:`AcquisitionStrategy` base
class, the concrete strategies, the diversity helpers, and the loop
orchestration interface. The :data:`STRATEGIES` registry maps strategy names to
their classes so they can be constructed from config.
"""

from __future__ import annotations

from typing import Any

from .base import AcquisitionStrategy, Pool, minmax_normalize
from .coreset import CoresetStrategy
from .diversity import kcenter_select, mmr_select, pairwise_distances
from .loop import (
    AlStepResult,
    not_implemented_infer_fn,
    not_implemented_train_fn,
    run_al_loop_step,
    run_budget_curve,
)
from .random import RandomStrategy
from .rehabal import DEFAULT_WEIGHTS, RehabALStrategy
from .temporal import TemporalStrategy
from .uncertainty import UncertaintyStrategy

__all__ = [
    "Pool",
    "AcquisitionStrategy",
    "minmax_normalize",
    "RandomStrategy",
    "UncertaintyStrategy",
    "CoresetStrategy",
    "TemporalStrategy",
    "RehabALStrategy",
    "DEFAULT_WEIGHTS",
    "kcenter_select",
    "mmr_select",
    "pairwise_distances",
    "run_al_loop_step",
    "run_budget_curve",
    "AlStepResult",
    "not_implemented_train_fn",
    "not_implemented_infer_fn",
    "STRATEGIES",
    "get_strategy",
    "register_strategy",
]


#: Registry mapping strategy name -> class.
STRATEGIES: dict[str, type[AcquisitionStrategy]] = {
    RandomStrategy.name: RandomStrategy,
    UncertaintyStrategy.name: UncertaintyStrategy,
    CoresetStrategy.name: CoresetStrategy,
    TemporalStrategy.name: TemporalStrategy,
    RehabALStrategy.name: RehabALStrategy,
}


def get_strategy(name: str, **kwargs: Any) -> AcquisitionStrategy:
    """Instantiate a registered strategy by name.

    Parameters
    ----------
    name:
        Registry key (see :data:`STRATEGIES`).
    **kwargs:
        Forwarded to the strategy constructor.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown strategy {name!r}; available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name](**kwargs)


def register_strategy(name: str, cls: type[AcquisitionStrategy]) -> None:
    """Register (or override) a strategy class under *name*."""
    if not issubclass(cls, AcquisitionStrategy):
        raise TypeError("cls must be an AcquisitionStrategy subclass")
    STRATEGIES[name] = cls
