"""CUSUM-style change point detector."""

from __future__ import annotations

import numpy as np

from ..base import BaseDetector


class ChangePointDetector(BaseDetector):
    """Detects abrupt level shifts using a cumulative sum statistic."""

    def __init__(self, threshold: float = 5.0, drift: float = 0.0, name: str = "changepoint") -> None:
        super().__init__(name=name, threshold=threshold)
        self.drift = drift
        self.reference_mean: float | None = None
        self.metadata.update({"drift": drift})

    def fit(self, X) -> None:
        arr = np.asarray(X, dtype=float)
        if arr.size:
            self.reference_mean = float(np.mean(arr))

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        mean = self.reference_mean if self.reference_mean is not None else float(np.mean(arr))
        pos_sum = 0.0
        neg_sum = 0.0
        max_pos = 0.0
        max_neg = 0.0
        for value in arr - mean:
            pos_sum = max(0.0, pos_sum + value - self.drift)
            neg_sum = min(0.0, neg_sum + value + self.drift)
            max_pos = max(max_pos, pos_sum)
            max_neg = min(max_neg, neg_sum)
        return float(max(max_pos, abs(max_neg)))
