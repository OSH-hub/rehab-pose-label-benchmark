"""Deterministic seeding across random / numpy / (optional) torch."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed"]


def set_seed(seed: int) -> int:
    """Seed Python ``random``, NumPy, and (if available) PyTorch.

    PyTorch is imported lazily and guarded so the test suite need not depend on
    it. The ``PYTHONHASHSEED`` environment variable is also set for
    reproducibility of hash-based ordering in subprocesses.

    Parameters
    ----------
    seed:
        The integer seed.

    Returns
    -------
    int
        The seed that was applied (echoed back for convenience).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is an optional, heavy dependency.
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch absent or failed to import
        pass

    return seed
