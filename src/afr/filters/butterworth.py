"""General-purpose Butterworth filter."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

from .base import BaseFilter


class ButterworthFilter(BaseFilter):
    """Configurable Butterworth filter supporting low/high/band-pass modes."""

    def __init__(
        self,
        cutoff: float | tuple[float, float] = 10.0,
        order: int = 4,
        mode: str = "lowpass",
        name: str = "butterworth",
    ) -> None:
        self.cutoff = cutoff
        self.order = order
        self.mode = mode.lower()
        self.name = name

    def apply(self, data: Sequence[float] | np.ndarray, sample_rate: float) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr

        if self.mode not in {"lowpass", "highpass", "bandpass"}:
            raise ValueError(f"Unsupported mode '{self.mode}' for ButterworthFilter")

        nyquist = sample_rate / 2.0
        cutoff = self.cutoff
        if isinstance(cutoff, tuple):
            low, high = cutoff
            high = min(high, nyquist * 0.999)
            low = max(low, 0.001)
            cutoff = (low, high)
        else:
            cutoff = min(float(cutoff), nyquist * 0.999)

        sos = signal.butter(
            self.order,
            cutoff,
            btype="band" if self.mode == "bandpass" else self.mode.replace("pass", ""),
            fs=sample_rate,
            output="sos",
        )
        return signal.sosfiltfilt(sos, arr)

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "cutoff": self.cutoff, "order": self.order, "mode": self.mode}
