"""Simple z-score based detector."""

from __future__ import annotations

import numpy as np

from ..base import BaseDetector


class StatisticalZScoreDetector(BaseDetector):
    """Flags windows whose max absolute z-score exceeds a threshold."""

    def __init__(self, threshold: float = 3.5, window: int | None = None, name: str = "zscore") -> None:
        super().__init__(name=name, threshold=threshold)
        self.window = window
        self.baseline_mean: float | None = None
        self.baseline_std: float | None = None
        self.metadata.update({"window": window})

    def fit(self, X) -> None:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return
        self.baseline_mean = float(np.mean(arr))
        self.baseline_std = float(np.std(arr) or 1e-9)

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0

        if self.window and arr.size > self.window:
            arr = arr[-self.window :]

        mean = self.baseline_mean if self.baseline_mean is not None else float(np.mean(arr))
        std = self.baseline_std if self.baseline_std is not None else float(np.std(arr) or 1e-9)
        z = np.abs((arr - mean) / std)
        return float(np.max(z))
