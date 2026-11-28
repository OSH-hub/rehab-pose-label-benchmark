from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed"]


def set_seed(seed: int) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch absent or failed to import
        pass

    return seed
