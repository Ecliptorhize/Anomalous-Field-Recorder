"""Bandpower-based anomaly detector."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from anomalous_field_recorder.signals import compute_bandpower
from ..base import BaseDetector


class SpectralBandpowerDetector(BaseDetector):
    """Flags windows whose relative bandpower exceeds a threshold."""

    def __init__(
        self,
        bands: Mapping[str, tuple[float, float]] | None = None,
        threshold: float | None = 0.35,
        target_bands: Iterable[str] | None = None,
        name: str = "bandpower",
        sample_rate: float | None = None,
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        self.bands = bands
        self.sample_rate = sample_rate or 1000.0
        self.target_bands = list(target_bands) if target_bands else ["beta_rel", "gamma_rel"]
        self.last_bandpower: Mapping[str, float] | None = None
        self.metadata.update({"bands": bands, "target_bands": self.target_bands})

    def update_context(self, sample_rate: float | None = None) -> None:
        if sample_rate:
            self.sample_rate = sample_rate

    def score(self, X: Sequence[float] | np.ndarray) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0 or self.sample_rate <= 0:
            return 0.0
        bandpower = compute_bandpower(arr, sample_rate=float(self.sample_rate), bands=self.bands)
        self.last_bandpower = bandpower
        if self.target_bands:
            scores = [bandpower.get(band, 0.0) for band in self.target_bands]
        else:
            scores = list(bandpower.values())
        return float(max(scores)) if scores else 0.0
