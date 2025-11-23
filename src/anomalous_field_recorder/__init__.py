"""Anomalous Field Recorder core package."""

from .config import load_experiment_config
from .pipeline import process_dataset, simulate_acquisition
from .reporting import generate_report

__all__ = [
    "load_experiment_config",
    "process_dataset",
    "simulate_acquisition",
    "generate_report",
]
