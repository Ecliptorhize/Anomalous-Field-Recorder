"""Compatibility layer for the Anomalous Field Recorder package.

This module exists to keep legacy imports working while development is in
progress. Preferred imports should use ``anomalous_field_recorder`` directly.
"""

from anomalous_field_recorder import (
    generate_report,
    load_experiment_config,
    process_dataset,
    simulate_acquisition,
)

__all__ = [
    "generate_report",
    "load_experiment_config",
    "process_dataset",
    "simulate_acquisition",
]
