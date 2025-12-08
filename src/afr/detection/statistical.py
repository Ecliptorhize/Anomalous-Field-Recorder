"""Statistical anomaly detectors used by the processing pipeline."""

from __future__ import annotations

import numpy as np

from afr.anomaly.base import BaseDetector


class ZScoreDetector(BaseDetector):
    """Flags windows whose z-score exceeds a configurable threshold."""

    def __init__(self, threshold: float = 3.5, window: int | None = None, name: str = "zscore") -> None:
        super().__init__(name=name, threshold=threshold)
        self.window = window
        self.metadata.update({"window": window})

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        if self.window and arr.size > self.window:
            arr = arr[-self.window :]
        mean = float(np.mean(arr))
        std = float(np.std(arr) or 1e-9)
        z = np.abs((arr - mean) / std)
        return float(np.max(z))


class RollingMeanVarianceDetector(BaseDetector):
    """Evaluates deviation of the most recent sample against rolling mean/std."""

    def __init__(
        self,
        window: int = 50,
        threshold: float = 3.0,
        min_std: float = 1e-6,
        name: str = "rolling_mean_variance",
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        self.window = window
        self.min_std = min_std
        self.metadata.update({"window": window, "min_std": min_std})

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        windowed = arr[-self.window :] if arr.size >= self.window else arr
        mean = float(np.mean(windowed))
        std = float(np.std(windowed))
        std = std if std > self.min_std else self.min_std
        last = float(arr[-1])
        return abs((last - mean) / std)


class MADDetector(BaseDetector):
    """Median Absolute Deviation detector for heavy-tailed noise."""

    def __init__(self, threshold: float = 3.5, name: str = "mad") -> None:
        super().__init__(name=name, threshold=threshold)

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad == 0:
            return 0.0
        modified_z = np.abs(arr - median) / (mad * 1.4826)
        return float(np.max(modified_z))
