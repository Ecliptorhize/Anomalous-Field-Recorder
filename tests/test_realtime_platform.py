from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from afr.anomaly.detectors import (
    AutoencoderDetector,
    ChangePointDetector,
    SpectralBandpowerDetector,
    StatisticalZScoreDetector,
)
from afr.anomaly.engine import AnomalyEngine
from afr.storage.sqlite import SQLiteBackend
from afr.storage.timescale import TimescaleBackend
from afr.streaming.service import StreamingService


def test_zscore_detector_flags_spike() -> None:
    detector = StatisticalZScoreDetector(threshold=2.0)
    window = [0.0] * 50 + [10.0]
    assert detector.predict(window)


def test_bandpower_detector_uses_relative_energy() -> None:
    sample_rate = 256.0
    t = np.linspace(0, 1, int(sample_rate), endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t)  # energy in alpha band
    detector = SpectralBandpowerDetector(threshold=0.1, target_bands=["alpha_rel"], sample_rate=sample_rate)
    score = detector.score(signal)
    assert score > 0.1


def test_autoencoder_detector_handles_missing_torch() -> None:
    detector = AutoencoderDetector(threshold=0.1, epochs=1, encoding_dim=2)
    score = detector.score([0.0, 0.1, -0.2, 0.3])
    assert isinstance(score, float)


def test_anomaly_engine_records_events(tmp_path: Path) -> None:
    storage = SQLiteBackend(tmp_path / "events.db")
    engine = AnomalyEngine(detectors=[StatisticalZScoreDetector(threshold=1.0), ChangePointDetector(threshold=0.5)], storage=storage)
    results = engine.evaluate_window([0, 0, 0, 5, 5, 5], sample_rate=100.0)
    assert any(r.is_anomaly for r in results)
    events = list(storage.fetch_recent_events())
    assert events


def test_streaming_service_runs_with_custom_source(tmp_path: Path) -> None:
    cfg = {
        "streaming": {"sample_rate": 100.0, "window_size": 8, "step_size": 4, "interval_ms": 1},
        "filters": {"band": [1.0, 40.0]},
        "engine": {"detectors": [{"type": "zscore", "threshold": 1.5}]},
    }

    batches = [[0.0 for _ in range(8)], [5.0 for _ in range(8)]]

    def source():
        for batch in batches:
            yield batch

    cfg["source"] = source
    storage = SQLiteBackend(tmp_path / "stream.db")
    service = StreamingService.create_from_config(cfg, storage=storage)
    results = service.run(max_batches=2)
    assert results
    assert list(storage.fetch_recent_events())


def test_timescale_backend_falls_back_without_connection() -> None:
    backend = TimescaleBackend("postgresql://localhost:1/afr")
    backend.store_event("dummy", 1.0, None, started_at=datetime(2024, 1, 1), metadata={})
    events = list(backend.fetch_recent_events())
    assert events
