"""Daemon-like streaming service orchestrating real-time anomaly detection."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Sequence

import yaml

import numpy as np

from ..alerting import AlertManager, SensorHealthMonitor, manager_from_config
from ..anomaly.engine import AnomalyEngine, DetectionResult
from ..registry.plugins.base import RegistryPlugin, NullRegistryPlugin
from ..storage.base import StorageBackend
from ..storage.sqlite import SQLiteBackend
from .buffering import WindowBuffer
from .filters import RealTimeFilterChain
from .processor import RealTimeProcessor
from .sources import build_source

logger = logging.getLogger(__name__)


class StreamingService:
    """Runs the streaming pipeline end-to-end."""

    def __init__(
        self,
        source: Callable[[], Iterable[Sequence[float]]],
        buffer: WindowBuffer,
        processor: RealTimeProcessor,
        engine: AnomalyEngine,
        *,
        interval_ms: int = 200,
        storage: StorageBackend | None = None,
        registry_plugin: RegistryPlugin | None = None,
        sample_rate: float = 1000.0,
        alert_manager: AlertManager | None = None,
        health_monitor: SensorHealthMonitor | None = None,
        backpressure_strategy: str = "sleep",
        config_path: str | None = None,
        raw_config: Mapping[str, object] | None = None,
    ) -> None:
        self.source = source
        self.buffer = buffer
        self.processor = processor
        self.engine = engine
        self.interval_ms = interval_ms
        self.storage = storage
        self.registry_plugin = registry_plugin or NullRegistryPlugin()
        self.sample_rate = float(sample_rate)
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
        self.backpressure_strategy = backpressure_strategy
        self._config_path = Path(config_path) if config_path else None
        self._config_mtime = self._config_path.stat().st_mtime if self._config_path and self._config_path.exists() else None
        self._raw_config = raw_config or {}
        self._running = False

    @classmethod
    def create_from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        storage: StorageBackend | None = None,
        registry_plugin: RegistryPlugin | None = None,
        config_path: str | None = None,
    ) -> "StreamingService":
        streaming_cfg = cfg.get("streaming", {})
        sample_rate = float(streaming_cfg.get("sample_rate", 256.0))  # type: ignore[arg-type]
        window_size = int(streaming_cfg.get("window_size", 256))  # type: ignore[arg-type]
        step_size = int(streaming_cfg.get("step_size", window_size // 2 or 1))  # type: ignore[arg-type]
        interval_ms = int(streaming_cfg.get("interval_ms", 200))  # type: ignore[arg-type]
        backpressure_strategy = str(streaming_cfg.get("backpressure_strategy", "sleep"))

        filters_cfg = cfg.get("filters", {})
        filter_chain = RealTimeFilterChain.from_config(filters_cfg, sample_rate=sample_rate)  # type: ignore[arg-type]
        processor = RealTimeProcessor(filter_chain)
        buffer = WindowBuffer(window_size=window_size, step_size=step_size)

        engine_cfg = cfg.get("engine", {"detectors": cfg.get("detectors", [])})
        engine = AnomalyEngine.from_config(engine_cfg, storage=storage, registry_plugin=registry_plugin)

        source = build_source(cfg.get("source"))
        alert_cfg = cfg.get("alerts", {})
        alert_manager = manager_from_config(alert_cfg) if alert_cfg else None

        health_cfg = cfg.get("health", {}) if cfg else {}
        health_monitor = SensorHealthMonitor(
            mean_drift_tol=float(health_cfg.get("mean_drift_tol", 5.0)),
            std_ceiling=float(health_cfg.get("std_ceiling", 5.0)),
            min_sample_rate=float(health_cfg.get("min_sample_rate", sample_rate)),
        )

        return cls(
            source=source,
            buffer=buffer,
            processor=processor,
            engine=engine,
            interval_ms=interval_ms,
            storage=storage,
            registry_plugin=registry_plugin,
            sample_rate=sample_rate,
            alert_manager=alert_manager,
            health_monitor=health_monitor,
            backpressure_strategy=backpressure_strategy,
            config_path=config_path,
            raw_config=cfg,
        )

    def process_samples(self, samples: Sequence[float]) -> List[DetectionResult]:
        windows = self.buffer.extend(samples)
        all_results: list[DetectionResult] = []
        for window in windows:
            processed = self.processor.process_window(window)
            results = self.engine.evaluate_window(
                processed.filtered,
                sample_rate=self.sample_rate,
                started_at=datetime.now(timezone.utc),
            )
            all_results.extend(results)
            health = self.health_monitor.check(processed.features, sample_rate=self.sample_rate) if self.health_monitor else None
            if self.alert_manager:
                self.alert_manager.evaluate(features=processed.features, detections=results, health=health)
        return all_results

    def run(self, *, max_batches: int | None = None, duration_s: float | None = None) -> List[DetectionResult]:
        """Run the streaming loop for the given duration or number of batches."""

        self._running = True
        results: list[DetectionResult] = []
        start = time.monotonic()
        for idx, samples in enumerate(self.source()):
            self._maybe_reload()
            if not self._running:
                break
            loop_start = time.monotonic()
            results.extend(self.process_samples(samples))
            self.registry_plugin.on_window(str(idx))
            if max_batches is not None and idx + 1 >= max_batches:
                break
            if duration_s is not None and (time.monotonic() - start) >= duration_s:
                break
            elapsed_ms = (time.monotonic() - loop_start) * 1000.0
            if elapsed_ms < self.interval_ms or self.backpressure_strategy == "sleep":
                time.sleep(max(self.interval_ms - elapsed_ms, 0.0) / 1000.0)
            elif self.backpressure_strategy == "drop":
                # drop accumulated buffer if we're too far behind
                self.buffer.reset()
                logger.warning("Backpressure: dropping buffered samples (elapsed_ms=%.2f)", elapsed_ms)
        self.registry_plugin.on_shutdown()
        return results

    def stop(self) -> None:
        self._running = False

    def _maybe_reload(self) -> None:
        if not self._config_path:
            return
        try:
            mtime = self._config_path.stat().st_mtime
        except FileNotFoundError:
            return
        if self._config_mtime and mtime <= self._config_mtime:
            return

        raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("Skipping reload: config not a mapping")
            return

        logger.info("Reloading streaming config from %s", self._config_path)
        self._config_mtime = mtime
        self._raw_config = raw
        updated = StreamingService.create_from_config(
            raw,
            storage=self.storage,
            registry_plugin=self.registry_plugin,
            config_path=str(self._config_path),
        )
        # swap runtime components
        self.buffer = updated.buffer
        self.processor = updated.processor
        self.engine = updated.engine
        self.sample_rate = updated.sample_rate
        self.interval_ms = updated.interval_ms
        self.backpressure_strategy = updated.backpressure_strategy
        self.alert_manager = updated.alert_manager
        self.health_monitor = updated.health_monitor
        self.source = updated.source


def build_sqlite_service(config: Mapping[str, object], path: str) -> StreamingService:
    """Helper to wire a default SQLite-backed service from config."""

    storage = SQLiteBackend(path)
    plugin = NullRegistryPlugin()
    return StreamingService.create_from_config(config, storage=storage, registry_plugin=plugin)
