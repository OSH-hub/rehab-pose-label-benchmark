from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["PoseModel", "Prediction"]


@dataclass
class Prediction:
    sample_id: str
    keypoints: np.ndarray
    bbox: Optional[list[float]] = None
    score: Optional[float] = None

    def __post_init__(self) -> None:
        kp = np.asarray(self.keypoints, dtype=np.float64)
        if kp.ndim != 2 or kp.shape[1] != 3:
            raise ValueError(
                f"Prediction.keypoints must have shape (K, 3) [x, y, score]; "
                f"got shape {kp.shape}"
            )
        self.keypoints = kp


class PoseModel(ABC):
    name: str = "abstract-pose-model"

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__}.train must be implemented by a concrete "
            "pose-model backend."
        )

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Sequence["Prediction"]:
        raise NotImplementedError(
            f"{type(self).__name__}.predict must be implemented by a concrete "
            "pose-model backend."
        )

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate must be implemented by a concrete "
            "pose-model backend."
        )

    @abstractmethod
    def extract_embeddings(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__}.extract_embeddings must be implemented by a "
            "concrete pose-model backend."
        )
