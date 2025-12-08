"""Shared filter interface and factory helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


class BaseFilter(ABC):
    """Common interface for all offline filters."""

    name: str = "filter"

    @abstractmethod
    def apply(self, data: Sequence[float] | np.ndarray, sample_rate: float) -> np.ndarray:
        """Return a filtered array."""

    def describe(self) -> Mapping[str, object]:
        return {"name": self.name}


@dataclass
class FilterChain:
    """Composable chain of BaseFilter instances."""

    filters: list[BaseFilter] = field(default_factory=list)

    def apply(self, data: Sequence[float] | np.ndarray, sample_rate: float) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if arr.size == 0:
            return arr
        for filt in self.filters:
            arr = np.asarray(filt.apply(arr, sample_rate), dtype=float)
        return arr

    @classmethod
    def from_config(cls, config: Iterable[Mapping[str, object]] | None, sample_rate: float) -> "FilterChain":
        from .bandpass import BandpassFilter
        from .butterworth import ButterworthFilter
        from .notch import NotchFilter
        from .smoothing import SmoothingFilter

        mapping = {
            "butterworth": ButterworthFilter,
            "notch": NotchFilter,
            "bandpass": BandpassFilter,
            "smoothing": SmoothingFilter,
        }

        filters: list[BaseFilter] = []
        for cfg in config or []:
            ftype = str(cfg.get("type") or cfg.get("name") or "").lower()  # type: ignore[union-attr]
            if not ftype:
                raise ValueError("Filter config entries require a 'type' or 'name' field")
            factory = mapping.get(ftype)
            if not factory:
                raise ValueError(f"Unknown filter type '{ftype}'. Available: {sorted(mapping)}")
            params = {k: v for k, v in cfg.items() if k not in {"type", "name"}}  # type: ignore[union-attr]
            filt = factory(**params) if params else factory()
            filters.append(filt)
        return cls(filters=filters)
