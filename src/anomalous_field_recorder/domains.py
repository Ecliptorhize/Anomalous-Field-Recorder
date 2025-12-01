"""Domain heuristics for multi-discipline experiment metadata."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


DOMAIN_PROFILES: Dict[str, Dict[str, Iterable[str]]] = {
    "field_engineering": {
        "required": ("site", "sample_rate"),
        "suggested": ("sensor_array", "calibration_date"),
    },
    "medical_imaging": {
        "required": ("modality", "patient_id", "exam_id"),
        "suggested": ("sequence", "safety_level"),
    },
    "clinical_lab": {
        "required": ("sample_type", "analyte", "lab_id"),
        "suggested": ("safety_level", "chain_of_custody"),
    },
    "chemistry_lab": {
        "required": ("instrument", "method", "lab_id"),
        "suggested": ("reagent_lot", "calibration_curve"),
    },
    "computational_neuroscience": {
        "required": ("subject_id", "session_id", "modality", "sample_rate"),
        "suggested": ("channels", "brain_region", "task"),
    },
}


def classify_domain(config: Mapping[str, Any]) -> str:
    """Infer a domain label from config fields."""

    declared = str(config.get("domain", "")).strip().lower()
    if declared in DOMAIN_PROFILES:
        return declared

    keys = {str(k).lower() for k in config.keys()}
    if {"modality", "patient_id"} & keys:
        return "medical_imaging"
    if {"sample_type", "analyte", "panel"} & keys:
        return "clinical_lab"
    if {"chromatography", "mass_spec", "method"} & keys:
        return "chemistry_lab"
    if {"eeg", "ephys", "lfp", "spike", "brain_region"} & keys or {"subject_id", "session_id"} <= keys:
        return "computational_neuroscience"
    return "field_engineering"


def summarize_domain(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a compact domain summary with missing fields and quality flags."""

    domain = classify_domain(config)
    profile = DOMAIN_PROFILES.get(domain, {})
    required = set(profile.get("required", ()))
    suggested = set(profile.get("suggested", ()))

    missing_required = sorted(key for key in required if key not in config)
    missing_suggested = sorted(key for key in suggested if key not in config)

    quality_flags = []
    if missing_required:
        quality_flags.append(f"missing required: {', '.join(missing_required)}")
    if missing_suggested:
        quality_flags.append(f"missing suggested: {', '.join(missing_suggested)}")
    if not quality_flags:
        quality_flags.append("all required metadata present")

    instrument = (
        config.get("instrument")
        or config.get("device")
        or config.get("sensor_array")
        or config.get("modality")
    )

    return {
        "domain": domain,
        "instrument": instrument,
        "missing_required": missing_required,
        "missing_suggested": missing_suggested,
        "quality_flags": quality_flags,
    }
