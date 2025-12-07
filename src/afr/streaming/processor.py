"""Real-time processing utilities for streaming windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from anomalous_field_recorder.signals import compute_bandpower, compute_spectral_entropy
from .filters import RealTimeFilterChain


@dataclass
class ProcessedWindow:
    window: list[float]
    filtered: list[float]
    features: Mapping[str, float]


class RealTimeProcessor:
    """Applies filters and derives lightweight features for each window."""

    def __init__(self, filter_chain: RealTimeFilterChain) -> None:
        self.filter_chain = filter_chain

    def process_window(self, window: Sequence[float]) -> ProcessedWindow:
        filtered = self.filter_chain.apply(window)
        arr = np.asarray(filtered, dtype=float)
        features = {
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "std": float(np.std(arr)) if arr.size else 0.0,
            "max": float(np.max(arr)) if arr.size else 0.0,
            "spectral_entropy": compute_spectral_entropy(arr, self.filter_chain.sample_rate) if arr.size else 0.0,
        }
        bp = compute_bandpower(arr, self.filter_chain.sample_rate) if arr.size else {}
        features.update({f"bandpower_{k}": v for k, v in bp.items()})
        return ProcessedWindow(window=list(window), filtered=filtered, features=features)

    def process_batch(self, windows: Iterable[Sequence[float]]) -> Tuple[list[ProcessedWindow], list[list[float]]]:
        processed: list[ProcessedWindow] = []
        filtered_windows: list[list[float]] = []
        for window in windows:
            pw = self.process_window(window)
            processed.append(pw)
            filtered_windows.append(pw.filtered)
        return processed, filtered_windows
