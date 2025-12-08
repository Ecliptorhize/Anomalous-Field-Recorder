"""Abstract detector definitions used by the anomaly engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class BaseDetector(ABC):
    """Base class for pluggable anomaly detectors."""

    name: str = "detector"
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: Sequence[float] | np.ndarray) -> None:
        """Optional fit step on historical data."""

    @abstractmethod
    def score(self, X: Sequence[float] | np.ndarray) -> float:
        """Return anomaly score; higher means more anomalous."""

    def predict(self, X: Sequence[float] | np.ndarray, *, score: float | None = None) -> bool:
        """Return True when the score exceeds the threshold."""

        value = self.score(X) if score is None else score
        if self.threshold is None:
            return bool(value > 0)
        return bool(value > self.threshold)

    def describe(self) -> Mapping[str, Any]:
        """Return serializable detector metadata."""

        return {"name": self.name, "threshold": self.threshold, **self.metadata}
