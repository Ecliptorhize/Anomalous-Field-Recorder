"""Anomalous Field Recorder core package."""

from importlib import metadata

from .config import load_experiment_config, validate_config, validate_config_file
from .domains import summarize_domain
from .pipeline import process_dataset, simulate_acquisition
from .registry import init_registry, list_runs, purge_runs, record_run
from .reporting import generate_report
from .signals import (
    apply_filters,
    compute_bandpower,
    compute_coherence,
    compute_event_locked_peak,
    compute_phase_locking_value,
    compute_spectral_entropy,
    compute_signal_metrics,
    compute_spectral_metrics,
    generate_synthetic_series,
    generate_multichannel_eeg,
    ingest_samples,
    score_anomalies,
)
from .service import create_app
from afr.anomaly.engine import AnomalyEngine, DetectionResult
from afr.streaming.service import StreamingService
from afr.storage.sqlite import SQLiteBackend
from afr.storage.timescale import TimescaleBackend

try:
    __version__ = metadata.version("anomalous-field-recorder")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.4.0"

__all__ = [
    "load_experiment_config",
    "validate_config",
    "validate_config_file",
    "generate_synthetic_series",
    "generate_multichannel_eeg",
    "compute_signal_metrics",
    "compute_spectral_metrics",
    "compute_bandpower",
    "compute_spectral_entropy",
    "compute_coherence",
    "compute_phase_locking_value",
    "compute_event_locked_peak",
    "apply_filters",
    "ingest_samples",
    "score_anomalies",
    "process_dataset",
    "simulate_acquisition",
    "generate_report",
    "init_registry",
    "record_run",
    "list_runs",
    "purge_runs",
    "create_app",
    "summarize_domain",
    "AnomalyEngine",
    "DetectionResult",
    "StreamingService",
    "SQLiteBackend",
    "TimescaleBackend",
    "__version__",
]
