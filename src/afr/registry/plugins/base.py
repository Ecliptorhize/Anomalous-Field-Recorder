"""Registry plugins allow routing detections into different sinks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from anomalous_field_recorder.registry import record_run

logger = logging.getLogger(__name__)


class RegistryPlugin:
    """Hook interface used by the anomaly engine and streaming service."""

    def on_start(self, run_id: str | None = None) -> None:
        ...

    def on_window(self, window_id: str | None = None) -> None:
        ...

    def on_detection(self, result, window) -> None:
        ...

    def on_shutdown(self) -> None:
        ...


class NullRegistryPlugin(RegistryPlugin):
    """No-op plugin for defaults."""


class LoggingRegistryPlugin(RegistryPlugin):
    """Log detections to the configured logger."""

    def __init__(self, level: int = logging.INFO) -> None:
        self.level = level

    def on_detection(self, result, window) -> None:  # pragma: no cover - simple logging path
        logger.log(
            self.level,
            "anomaly_detected",
            extra={
                "detector": result.detector,
                "score": result.score,
                "threshold": result.threshold,
                "size": len(window),
            },
        )


class SqliteRegistryPlugin(RegistryPlugin):
    """Persist detections into the existing AFR registry table."""

    def __init__(self, db_path: str | Path, domain: str = "unknown") -> None:
        self.db_path = Path(db_path)
        self.domain = domain

    def on_detection(self, result, window) -> None:
        record_run(
            self.db_path,
            kind="anomaly",
            location=f"window:{len(window)}",
            status="triggered",
            domain=self.domain,
        )


class CompositeRegistryPlugin(RegistryPlugin):
    """Fan-out to multiple plugins."""

    def __init__(self, plugins: Iterable[RegistryPlugin]) -> None:
        self.plugins = list(plugins)

    def on_start(self, run_id: str | None = None) -> None:
        for plugin in self.plugins:
            plugin.on_start(run_id)

    def on_window(self, window_id: str | None = None) -> None:
        for plugin in self.plugins:
            plugin.on_window(window_id)

    def on_detection(self, result, window) -> None:
        for plugin in self.plugins:
            plugin.on_detection(result, window)

    def on_shutdown(self) -> None:
        for plugin in self.plugins:
            plugin.on_shutdown()
