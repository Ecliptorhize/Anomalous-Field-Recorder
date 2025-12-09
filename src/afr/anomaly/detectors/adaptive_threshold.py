"""Adaptive threshold detector with exponential smoothing."""

from __future__ import annotations

import numpy as np

from ..base import BaseDetector


class AdaptiveThresholdDetector(BaseDetector):
    """Adapts threshold based on running mean/std (EWMA)."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        k_sigma: float = 3.0,
        min_std: float = 1e-6,
        name: str = "adaptive_threshold",
    ) -> None:
        super().__init__(name=name, threshold=None)
        self.alpha = alpha
        self.k_sigma = k_sigma
        self.min_std = min_std
        self._mean: float | None = None
        self._var: float | None = None
        self.metadata.update({"alpha": alpha, "k_sigma": k_sigma})

    def update_context(self, *, sample_rate: float | None = None) -> None:
        if sample_rate:
            self.metadata["sample_rate"] = sample_rate

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        current_mean = float(np.mean(arr))
        current_var = float(np.var(arr))

        if self._mean is None:
            self._mean = current_mean
            self._var = current_var
        else:
            self._mean = self.alpha * current_mean + (1 - self.alpha) * self._mean
            self._var = self.alpha * current_var + (1 - self.alpha) * self._var

        std = float(np.sqrt(max(self._var or 0.0, self.min_std)))
        z = abs(current_mean - (self._mean or current_mean)) / std if std else 0.0
        self.threshold = self.k_sigma
        return z
