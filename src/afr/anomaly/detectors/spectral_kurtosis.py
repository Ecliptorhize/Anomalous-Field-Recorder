"""Spectral kurtosis detector to catch transient bursts."""

from __future__ import annotations

import numpy as np

from ..base import BaseDetector


class SpectralKurtosisDetector(BaseDetector):
    """Uses kurtosis of the power spectrum as an impulse indicator."""

    def __init__(self, threshold: float = 5.0, name: str = "spectral_kurtosis") -> None:
        super().__init__(name=name, threshold=threshold)

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        spectrum = np.abs(np.fft.rfft(arr))
        if spectrum.size == 0:
            return 0.0
        mean = float(np.mean(spectrum))
        std = float(np.std(spectrum) or 1.0)
        z = (spectrum - mean) / std
        kurtosis = float(np.mean(z**4))
        return kurtosis
