"""Lightweight smoothing filters."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseFilter


class SmoothingFilter(BaseFilter):
    """Moving-average or exponential smoothing."""

    def __init__(
        self,
        window: int = 5,
        alpha: float | None = None,
        method: str = "moving_average",
        name: str = "smoothing",
    ) -> None:
        self.window = window
        self.alpha = alpha
        self.method = method
        self.name = name

    def apply(self, data: Sequence[float] | np.ndarray, sample_rate: float) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0 or self.window <= 1:
            return arr
        if self.method == "ema" and self.alpha is not None:
            smoothed = np.zeros_like(arr)
            smoothed[0] = arr[0]
            alpha = float(self.alpha)
            for i in range(1, len(arr)):
                smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
            return smoothed

        window = max(int(self.window), 1)
        kernel = np.ones(window) / float(window)
        return np.convolve(arr, kernel, mode="same")

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "window": self.window, "alpha": self.alpha, "method": self.method}
