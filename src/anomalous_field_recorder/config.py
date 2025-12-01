"""Configuration loading and validation utilities for experiment definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .domains import classify_domain

SimpleValue = str | int | float | bool | None

DOMAIN_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "field_engineering": {
        "required": {"site": str, "sample_rate": (int, float)},
        "optional": {"sensor_array": str, "calibration_date": str, "instrument": str},
        "defaults": {"sample_rate": 1000},
        "constraints": {"sample_rate": {"min": 1, "max": 1_000_000}},
    },
    "medical_imaging": {
        "required": {"patient_id": str, "exam_id": str, "modality": str},
        "optional": {"sequence": str, "safety_level": str},
        "defaults": {"safety_level": "mri-safe"},
    },
    "clinical_lab": {
        "required": {"sample_type": str, "analyte": str, "lab_id": str},
        "optional": {"panel": str, "safety_level": str, "instrument": str},
        "defaults": {"panel": "unspecified"},
    },
    "chemistry_lab": {
        "required": {"instrument": str, "method": str, "lab_id": str},
        "optional": {"reagent_lot": str, "calibration_curve": str, "chromatography": str},
        "defaults": {"calibration_curve": "not-provided"},
    },
}


@dataclass
class ValidationResult:
    domain: str
    errors: list[str]
    warnings: list[str]
    normalized: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "errors": self.errors,
            "warnings": self.warnings,
            "normalized": self.normalized,
        }


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


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(_type_name(e) for e in expected)
    if isinstance(expected, type):
        return expected.__name__
    return str(expected)


def _apply_defaults(config: Mapping[str, Any], domain: str) -> Dict[str, Any]:
    normalized = dict(config)
    schema = DOMAIN_SCHEMAS.get(domain, {})
    for key, value in schema.get("defaults", {}).items():
        normalized.setdefault(key, value)
    if "domain" not in normalized:
        normalized["domain"] = domain
    return normalized


def validate_config(config: Mapping[str, Any]) -> ValidationResult:
    """Validate a configuration mapping against domain-aware schema.

    Validation does not mutate the original config. It applies defaults where
    possible and reports errors for missing or mistyped required fields while
    emitting warnings for optional fields that are present but malformed.
    """

    domain = classify_domain(config)
    schema = DOMAIN_SCHEMAS.get(domain, {})
    errors: list[str] = []
    warnings: list[str] = []

    normalized = _apply_defaults(config, domain)

    required = schema.get("required", {})
    for field, expected_type in required.items():
        if field not in normalized:
            errors.append(f"Missing required field '{field}' for domain '{domain}'")
            continue
        if not isinstance(normalized[field], expected_type):
            errors.append(
                f"Field '{field}' should be of type {_type_name(expected_type)} "
                f"(got {type(normalized[field]).__name__})"
            )

    optional = schema.get("optional", {})
    for field, expected_type in optional.items():
        if field in normalized and not isinstance(normalized[field], expected_type):
            warnings.append(
                f"Field '{field}' should be of type {_type_name(expected_type)} "
                f"(got {type(normalized[field]).__name__})"
            )

    constraints = schema.get("constraints", {})
    for field, constraint in constraints.items():
        if field not in normalized or not isinstance(normalized[field], (int, float)):
            continue
        value = float(normalized[field])
        min_value = constraint.get("min")
        max_value = constraint.get("max")
        if min_value is not None and value < min_value:
            errors.append(f"Field '{field}' must be >= {min_value} (got {value})")
        if max_value is not None and value > max_value:
            warnings.append(f"Field '{field}' is unusually high ({value} > {max_value})")

    return ValidationResult(
        domain=domain,
        errors=errors,
        warnings=warnings,
        normalized=normalized,
    )


def validate_config_file(path: str | Path) -> ValidationResult:
    """Load and validate a configuration file."""
    config = load_experiment_config(path)
    return validate_config(config)


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
