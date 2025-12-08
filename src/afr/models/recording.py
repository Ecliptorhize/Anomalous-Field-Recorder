"""Recording model representing a loaded dataset."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .metadata import RecordingMetadata


class Recording(BaseModel):
    """Structured representation of a time-series recording."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    sample_rate: float
    values: List[float]
    timestamps: Optional[List[datetime]] = None
    metadata: Optional[RecordingMetadata] = None
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("sample_rate")
    @classmethod
    def _positive_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return float(value)

    @field_validator("timestamps")
    @classmethod
    def _match_lengths(cls, timestamps: Optional[List[datetime]], info) -> Optional[List[datetime]]:
        values = info.data.get("values") or []
        if timestamps and len(timestamps) != len(values):
            raise ValueError("timestamps length must match values length")
        return timestamps

    def window(self, start: int, length: int) -> np.ndarray:
        arr = np.asarray(self.values, dtype=float)
        end = min(start + length, arr.size)
        return arr[start:end]

    def timestamp_for_index(self, index: int) -> datetime:
        if self.timestamps and 0 <= index < len(self.timestamps):
            return self.timestamps[index]
        return self.start_time + timedelta(seconds=index / self.sample_rate)

    def duration_seconds(self) -> float:
        if not self.values:
            return 0.0
        return len(self.values) / float(self.sample_rate)
