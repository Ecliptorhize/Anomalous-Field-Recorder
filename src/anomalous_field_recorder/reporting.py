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

    output_path = Path(output_path) if output_path else processed_dir / "report.md"
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path
