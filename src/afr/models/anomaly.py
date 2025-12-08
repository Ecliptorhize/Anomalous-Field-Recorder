"""Anomaly representations for downstream reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class AnomalyEvent(BaseModel):
    """Structured anomaly event with detector context."""

    model_config = ConfigDict(extra="allow")

    index: int
    timestamp: datetime
    detector: str
    score: float
    threshold: Optional[float] = None
    confidence: Optional[float] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    def short_label(self) -> str:
        return f"{self.detector}@{self.index}"
