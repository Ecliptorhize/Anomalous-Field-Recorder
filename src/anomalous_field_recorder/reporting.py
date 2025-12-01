"""Reporting helpers for processed datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .pipeline import DEFAULT_SUMMARY_FILE


def generate_report(processed_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Generate a human-readable report from a processed dataset.

    The report is a Markdown file that summarizes configuration keys and basic
    statistics. If ``output_path`` is not provided, ``report.md`` within the
    processed directory is used.
    """

    processed_dir = Path(processed_dir)
    summary_path = processed_dir / DEFAULT_SUMMARY_FILE
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary: Dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    report_lines = [
        "# Anomalous Field Recorder Report",
        "",
        f"Source: {summary.get('source', 'unknown')}",
        f"Status: {summary.get('status', 'unknown')}",
        f"Records: {summary.get('records', 0)}",
        f"Domain: {summary.get('domain', 'field_engineering')}",
        f"Instrument: {summary.get('instrument', 'unspecified')}",
        "",
        "## Configuration Keys",
    ]

    config_keys = summary.get("config_keys", [])
    if config_keys:
        report_lines.extend(f"- {key}" for key in config_keys)
    else:
        report_lines.append("(none recorded)")

    report_lines.append("")
    report_lines.append("## Quality Flags")
    quality_flags = summary.get("quality_flags") or []
    if quality_flags:
        report_lines.extend(f"- {flag}" for flag in quality_flags)
    else:
        report_lines.append("(no quality flags)")

    report_lines.append("")
    report_lines.append("## Signal Metrics")
    metrics = summary.get("metrics") or {}
    if metrics:
        for key, value in metrics.items():
            report_lines.append(f"- {key}: {value}")
    else:
        report_lines.append("(no metrics)")

    report_lines.append("")
    report_lines.append("## Spectral Summary")
    spectral = summary.get("spectral") or {}
    if spectral:
        for key, value in spectral.items():
            report_lines.append(f"- {key}: {value}")
    else:
        report_lines.append("(no spectral data)")

    report_lines.append("")
    report_lines.append("## Bandpower")
    bandpower = summary.get("bandpower") or {}
    if bandpower:
        for key, value in bandpower.items():
            report_lines.append(f"- {key}: {value}")
    else:
        report_lines.append("(no bandpower data)")

    report_lines.append("")
    report_lines.append("## Anomalies")
    anomalies = summary.get("anomalies") or {}
    if anomalies:
        report_lines.append(f"- count: {anomalies.get('count', 0)}")
        report_lines.append(f"- threshold: {anomalies.get('threshold', 'n/a')}")
        report_lines.append(f"- indices: {anomalies.get('indices', [])}")
    else:
        report_lines.append("(no anomalies detected)")

    filters = summary.get("filters") or {}
    report_lines.append("")
    report_lines.append("## Filters Applied")
    if filters.get("band") or filters.get("notch"):
        report_lines.append(f"- band: {filters.get('band')}")
        report_lines.append(f"- notch: {filters.get('notch')}")
    else:
        report_lines.append("(no filters)")

    output_path = Path(output_path) if output_path else processed_dir / "report.md"
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path
