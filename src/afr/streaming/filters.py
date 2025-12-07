"""Filter chain utilities for real-time processing."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import numpy as np
from scipy import signal


class RealTimeFilterChain:
    """Composable real-time filter chain supporting bandpass/notch/moving RMS."""

    def __init__(
        self,
        sample_rate: float,
        band: tuple[float, float] | None = None,
        notch: float | None = None,
        rms_window: int | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.band = band
        self.notch = notch
        self.rms_window = rms_window

    @classmethod
    def from_config(cls, cfg: Mapping[str, object], sample_rate: float) -> "RealTimeFilterChain":
        band = tuple(cfg["band"]) if isinstance(cfg.get("band"), (list, tuple)) else None  # type: ignore[arg-type]
        notch = float(cfg["notch"]) if cfg.get("notch") is not None else None  # type: ignore[arg-type]
        rms_window = int(cfg["rms_window"]) if cfg.get("rms_window") is not None else None  # type: ignore[arg-type]
        return cls(sample_rate=sample_rate, band=band, notch=notch, rms_window=rms_window)

    def apply(self, samples: Sequence[float]) -> List[float]:
        if not samples:
            return []
        arr = np.asarray(samples, dtype=float)
        if self.band:
            low, high = self.band
            nyquist = self.sample_rate / 2.0
            high = min(high, nyquist - 1e-6)
            low = max(low, 0.001)
            if high > low:
                sos = signal.butter(4, [low, high], btype="band", fs=self.sample_rate, output="sos")
                arr = signal.sosfilt(sos, arr)

        if self.notch and self.notch < (self.sample_rate / 2.0):
            b, a = signal.iirnotch(self.notch, 30, self.sample_rate)
            arr = signal.lfilter(b, a, arr)

        if self.rms_window and self.rms_window > 1:
            window = np.ones(self.rms_window) / float(self.rms_window)
            squared = arr**2
            rms = np.sqrt(np.convolve(squared, window, mode="same"))
            arr = rms

        return arr.tolist()
