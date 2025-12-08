"""Recording metadata definitions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecordingMetadata(BaseModel):
    """Captures contextual information about a dataset."""

    model_config = ConfigDict(extra="allow")

    source: str
    sample_rate: float
    sensor_type: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_rate")
    @classmethod
    def _ensure_positive_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("sample_rate must be positive")
        return float(value)
