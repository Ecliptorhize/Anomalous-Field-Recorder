"""Lightweight matrix-profile-style discord detector."""

from __future__ import annotations

import math
import numpy as np

from ..base import BaseDetector


class MatrixProfileDetector(BaseDetector):
    """Flags subsequences that are dissimilar from the rest of the window."""

    def __init__(self, m: int = 32, threshold: float = 5.0, name: str = "matrix_profile") -> None:
        super().__init__(name=name, threshold=threshold)
        self.m = int(m)
        self.metadata.update({"m": m})

    def _znorm(self, seq: np.ndarray) -> np.ndarray:
        mean = seq.mean()
        std = seq.std() or 1.0
        return (seq - mean) / std

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        n = arr.size
        m = self.m
        if n < 2 * m or m <= 2:
            return 0.0

        profiles = []
        for i in range(0, n - m + 1):
            s1 = self._znorm(arr[i : i + m])
            best = math.inf
            for j in range(0, n - m + 1):
                if abs(i - j) < m:  # avoid trivial matches
                    continue
                s2 = self._znorm(arr[j : j + m])
                dist = float(np.linalg.norm(s1 - s2))
                if dist < best:
                    best = dist
            if best is not math.inf:
                profiles.append(best)

        if not profiles:
            return 0.0
        return float(max(profiles))
