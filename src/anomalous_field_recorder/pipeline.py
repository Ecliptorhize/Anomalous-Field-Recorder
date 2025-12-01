"""Data acquisition and processing helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import load_experiment_config, validate_config
from .domains import summarize_domain
from .logging_utils import log_event
from .registry import record_run
from .signals import (
    apply_filters,
    compute_signal_metrics,
    compute_bandpower,
    compute_spectral_metrics,
    generate_synthetic_series,
    score_anomalies,
)


DEFAULT_METADATA_FILE = "metadata.json"
DEFAULT_SUMMARY_FILE = "summary.json"
logger = logging.getLogger(__name__)


def simulate_acquisition(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    generate_samples: bool = True,
    duration_s: float = 1.0,
    registry_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Simulate acquisition by recording config and timestamps.

    The function creates the ``output_dir`` if needed and writes a ``metadata.json``
    file that mirrors the loaded configuration along with a timestamp. This keeps
    the repository usable in environments without actual hardware attached.
    """

    config = load_experiment_config(config_path)
    validation = validate_config(config)
    sample_rate = float(config.get("sample_rate", 1_000))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_profile = summarize_domain(config)

    samples: list[float] = []
    if generate_samples:
        samples = generate_synthetic_series(duration_s=duration_s, sample_rate=sample_rate)

    metadata = {
        "config": config,
        "status": "acquired",
        "samples": samples,
        "domain_profile": domain_profile,
        "validation": validation.as_dict(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = output_dir / DEFAULT_METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_event(
        logger,
        "acquisition_complete",
        output=str(output_dir),
        sample_rate=sample_rate,
        samples=len(samples),
        domain=domain_profile["domain"],
    )
    if registry_path:
        record_run(registry_path, "acquire", str(output_dir), metadata["status"], domain_profile["domain"])
    return metadata


def process_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    band: tuple[float, float] | None = None,
    notch: float | None = None,
    registry_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Process a raw dataset into a lightweight summary.

    The processor looks for ``metadata.json`` in ``raw_dir`` to extract
    configuration details. It writes a ``summary.json`` to ``processed_dir``
    containing a handful of derived statistics that downstream reporting can use.
    """

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir / DEFAULT_METADATA_FILE
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    config = metadata.get("config", {}) if isinstance(metadata.get("config", {}), dict) else {}
    samples = metadata.get("samples", []) if isinstance(metadata.get("samples", []), list) else []
    sample_rate = float(config.get("sample_rate", 1_000))

    domain_profile = summarize_domain(config)

    filtered_samples = apply_filters(samples, sample_rate, band=band, notch=notch) if samples else []
    metrics = compute_signal_metrics(filtered_samples or samples)
    spectral = compute_spectral_metrics(filtered_samples or samples, sample_rate)
    anomalies = score_anomalies(filtered_samples or samples)
    bandpower = compute_bandpower(filtered_samples or samples, sample_rate)

    summary = {
        "source": str(raw_dir.resolve()),
        "records": len(samples),
        "config_keys": sorted(config.keys()),
        "status": metadata.get("status", "unknown"),
        "domain": domain_profile["domain"],
        "instrument": domain_profile["instrument"],
        "quality_flags": domain_profile["quality_flags"],
        "metrics": metrics,
        "spectral": spectral,
        "anomalies": anomalies,
        "bandpower": bandpower,
        "filters": {"band": band, "notch": notch},
    }

    summary_path = processed_dir / DEFAULT_SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_event(
        logger,
        "processing_complete",
        source=str(raw_dir),
        processed=str(processed_dir),
        samples=len(samples),
        domain=domain_profile["domain"],
    )
    if registry_path:
        record_run(registry_path, "process", str(processed_dir), summary["status"], summary["domain"])
    return summary
