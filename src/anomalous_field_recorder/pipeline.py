"""Data acquisition and processing helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import load_experiment_config


DEFAULT_METADATA_FILE = "metadata.json"
DEFAULT_SUMMARY_FILE = "summary.json"


def simulate_acquisition(config_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Simulate acquisition by recording config and timestamps.

    The function creates the ``output_dir`` if needed and writes a ``metadata.json``
    file that mirrors the loaded configuration along with a timestamp. This keeps
    the repository usable in environments without actual hardware attached.
    """

    config = load_experiment_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": config,
        "status": "acquired",
        "samples": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = output_dir / DEFAULT_METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def process_dataset(raw_dir: str | Path, processed_dir: str | Path) -> Dict[str, Any]:
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

    summary = {
        "source": str(raw_dir.resolve()),
        "records": len(samples),
        "config_keys": sorted(config.keys()),
        "status": metadata.get("status", "unknown"),
    }

    summary_path = processed_dir / DEFAULT_SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
