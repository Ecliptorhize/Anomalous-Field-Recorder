"""Daemon-like streaming service orchestrating real-time anomaly detection."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Mapping, Sequence

import numpy as np

from ..anomaly.engine import AnomalyEngine, DetectionResult
from ..registry.plugins.base import RegistryPlugin, NullRegistryPlugin
from ..storage.base import StorageBackend
from ..storage.sqlite import SQLiteBackend
from .buffering import WindowBuffer
from .filters import RealTimeFilterChain
from .processor import RealTimeProcessor

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
    ) -> None:
        self.source = source
        self.buffer = buffer
        self.processor = processor
        self.engine = engine
        self.interval_ms = interval_ms
        self.storage = storage
        self.registry_plugin = registry_plugin or NullRegistryPlugin()
        self.sample_rate = float(sample_rate)
        self._running = False

    @classmethod
    def create_from_config(
        cls,
        cfg: Mapping[str, object],
        *,
        storage: StorageBackend | None = None,
        registry_plugin: RegistryPlugin | None = None,
    ) -> "StreamingService":
        streaming_cfg = cfg.get("streaming", {})
        sample_rate = float(streaming_cfg.get("sample_rate", 256.0))  # type: ignore[arg-type]
        window_size = int(streaming_cfg.get("window_size", 256))  # type: ignore[arg-type]
        step_size = int(streaming_cfg.get("step_size", window_size // 2 or 1))  # type: ignore[arg-type]
        interval_ms = int(streaming_cfg.get("interval_ms", 200))  # type: ignore[arg-type]

        filters_cfg = cfg.get("filters", {})
        filter_chain = RealTimeFilterChain.from_config(filters_cfg, sample_rate=sample_rate)  # type: ignore[arg-type]
        processor = RealTimeProcessor(filter_chain)
        buffer = WindowBuffer(window_size=window_size, step_size=step_size)

        engine_cfg = cfg.get("engine", {"detectors": cfg.get("detectors", [])})
        engine = AnomalyEngine.from_config(engine_cfg, storage=storage, registry_plugin=registry_plugin)

        source = cls._build_source(cfg.get("source"))
        return cls(
            source=source,
            buffer=buffer,
            processor=processor,
            engine=engine,
            interval_ms=interval_ms,
            storage=storage,
            registry_plugin=registry_plugin,
            sample_rate=sample_rate,
        )

    @staticmethod
    def _build_source(source_cfg: object) -> Callable[[], Iterable[Sequence[float]]]:
        if callable(source_cfg):
            return source_cfg  # type: ignore[return-value]

        def simulator() -> Iterable[Sequence[float]]:
            rng = np.random.default_rng()
            while True:
                base = rng.normal(scale=0.1, size=128).tolist()
                if rng.random() < 0.1:
                    # inject a burst to emulate anomalies
                    base = [x + rng.normal(scale=2.0) for x in base]
                yield base

        return simulator

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
        return all_results

    def run(self, *, max_batches: int | None = None, duration_s: float | None = None) -> List[DetectionResult]:
        """Run the streaming loop for the given duration or number of batches."""

        self._running = True
        results: list[DetectionResult] = []
        start = time.monotonic()
        for idx, samples in enumerate(self.source()):
            if not self._running:
                break
            results.extend(self.process_samples(samples))
            self.registry_plugin.on_window(str(idx))
            if max_batches is not None and idx + 1 >= max_batches:
                break
            if duration_s is not None and (time.monotonic() - start) >= duration_s:
                break
            time.sleep(self.interval_ms / 1000.0)
        self.registry_plugin.on_shutdown()
        return results

    def stop(self) -> None:
        self._running = False


def build_sqlite_service(config: Mapping[str, object], path: str) -> StreamingService:
    """Helper to wire a default SQLite-backed service from config."""

    storage = SQLiteBackend(path)
    plugin = NullRegistryPlugin()
    return StreamingService.create_from_config(config, storage=storage, registry_plugin=plugin)
