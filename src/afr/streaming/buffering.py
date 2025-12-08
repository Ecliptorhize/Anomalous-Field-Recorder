"""Window buffering utilities for streaming pipelines."""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Sequence


class WindowBuffer:
    """Rolling buffer that emits fixed-size windows with optional overlap."""

    def __init__(self, window_size: int, step_size: int | None = None) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.step_size = int(step_size) if step_size else self.window_size
        self._buffer: deque[float] = deque()

    def extend(self, samples: Iterable[float]) -> List[List[float]]:
        self._buffer.extend(samples)
        windows: list[list[float]] = []
        while len(self._buffer) >= self.window_size:
            window = list(x for idx, x in zip(range(self.window_size), self._buffer))
            windows.append(window)
            for _ in range(self.step_size):
                if not self._buffer:
                    break
                self._buffer.popleft()
        return windows

    def reset(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
