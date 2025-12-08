"""Simple bandpass filter wrapper."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import signal

from .base import BaseFilter


class BandpassFilter(BaseFilter):
    """Narrow convenience wrapper for Butterworth bandpass."""

    def __init__(self, band: Tuple[float, float] = (1.0, 40.0), order: int = 4, name: str = "bandpass") -> None:
        self.band = band
        self.order = order
        self.name = name

    def apply(self, data: Sequence[float] | np.ndarray, sample_rate: float) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        low, high = self.band
        nyquist = sample_rate / 2.0
        high = min(high, nyquist * 0.999)
        low = max(low, 0.001)
        sos = signal.butter(self.order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        return signal.sosfiltfilt(sos, arr)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "band": self.band, "order": self.order}
