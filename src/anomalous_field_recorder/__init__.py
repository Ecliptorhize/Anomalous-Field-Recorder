"""Anomalous Field Recorder core package."""

from importlib import metadata

from .config import load_experiment_config
from .domains import summarize_domain
from .pipeline import process_dataset, simulate_acquisition
from .reporting import generate_report

try:
    __version__ = metadata.version("anomalous-field-recorder")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.2.0"

__all__ = [
    "load_experiment_config",
    "process_dataset",
    "simulate_acquisition",
    "generate_report",
    "summarize_domain",
    "__version__",
]
