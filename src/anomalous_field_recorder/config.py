"""Configuration loading utilities for experiment definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SimpleValue = str | int | float | bool | None


def _coerce_value(value: str) -> SimpleValue:
    """Attempt to coerce a string into a basic Python type."""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_lightweight_yaml(text: str) -> Dict[str, SimpleValue]:
    """Parse a minimal subset of YAML (key/value pairs) without dependencies.

    The function intentionally supports a small YAML fragment to avoid
    introducing external dependencies when PyYAML is unavailable. Nested
    structures are not supported, which is sufficient for simple experiment
    metadata used in tests and sample configurations.
    """

    data: Dict[str, SimpleValue] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid line in configuration: {line}")
        key, raw_value = stripped.split(":", 1)
        data[key.strip()] = _coerce_value(raw_value.strip())
    return data


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML or JSON.

    The function prefers PyYAML when installed but gracefully falls back to a
    lightweight parser that understands simple ``key: value`` pairs. JSON files
    are also supported for users that prefer a stricter format.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Try PyYAML if available
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)  # type: ignore[arg-type]
        if isinstance(loaded, dict):
            return loaded
    except ModuleNotFoundError:
        pass

    # Try JSON next
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_lightweight_yaml(text)
