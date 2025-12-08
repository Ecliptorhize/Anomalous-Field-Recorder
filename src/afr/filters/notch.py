"""Notch filter for line noise removal."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

from .base import BaseFilter


class NotchFilter(BaseFilter):
    """Removes a narrow frequency component (e.g., mains hum)."""

    def __init__(self, frequency: float = 60.0, quality: float = 30.0, name: str = "notch") -> None:
        self.frequency = frequency
        self.quality = quality
        self.name = name

    def apply(self, data: Sequence[float] | np.ndarray, sample_rate: float) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        if self.frequency >= (sample_rate / 2.0):
            return arr
        b, a = signal.iirnotch(self.frequency, self.quality, sample_rate)
        return signal.lfilter(b, a, arr)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "frequency": self.frequency, "quality": self.quality}
