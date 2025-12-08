"""Storage abstraction for streaming artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Mapping, Sequence

import numpy as np


class StorageBackend(ABC):
    """Abstract storage backend for windows and anomaly events."""

    @abstractmethod
    def store_window(self, window: Sequence[float] | np.ndarray, sample_rate: float, started_at: datetime) -> None:
        ...

    @abstractmethod
    def store_event(
        self,
        detector: str,
        score: float,
        threshold: float | None,
        started_at: datetime,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def fetch_recent_events(self, limit: int = 50) -> Iterable[Mapping[str, object]]:
        ...

    def close(self) -> None:
        ...
