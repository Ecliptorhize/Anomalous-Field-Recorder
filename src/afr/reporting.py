"""Reporting utilities that emit JSON, Markdown, and PDF artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from matplotlib.backends.backend_pdf import PdfPages

from .models import AnomalyEvent, Recording

_MARKDOWN_TEMPLATE = """# {{ title }}

- Source: {{ recording.metadata.source if recording.metadata else "unknown" }}
- Sample rate: {{ recording.sample_rate }} Hz
- Duration: {{ summary.duration_s | round(3) }} seconds
- Samples: {{ summary.n_samples }}
- Anomalies: {{ summary.n_anomalies }}

## Filters
{% if filters %}
{% for f in filters %}- {{ f }}
{% endfor %}
{% else %}- none
{% endif %}

## Detectors
{% if detectors %}
{% for d in detectors %}- {{ d.get("name", d) }} (threshold={{ d.get("threshold", "n/a") }})
{% endfor %}
{% else %}- none
{% endif %}

## Anomalies
{% if anomalies %}
{% for a in anomalies %}
- {{ a.timestamp.isoformat() }} | {{ a.detector }} | score={{ "%.3f"|format(a.score) }} threshold={{ a.threshold }}
{% endfor %}
{% else %}
No anomalies detected for the configured thresholds.
{% endif %}
"""


def render_markdown_report(context: dict[str, Any], output_path: str | Path) -> Path:
    """Render a Markdown report using the built-in template."""

    template = Template(_MARKDOWN_TEMPLATE)
    rendered = template.render(**context)
    output = Path(output_path)
    output.write_text(rendered, encoding="utf-8")
    return output


def write_pdf_report(
    recording: Recording,
    raw: Sequence[float],
    filtered: Sequence[float],
    anomalies: Iterable[AnomalyEvent],
    output_path: str | Path,
) -> Path:
    """Build a PDF with time-series plots and anomaly markers."""

    output = Path(output_path)
    anomalies = list(anomalies)
    times = np.arange(len(raw)) / float(recording.sample_rate)

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, raw, label="raw", color="#8c564b", alpha=0.6)
        ax.plot(times, filtered, label="filtered", color="#1f77b4", linewidth=1.2)
        for event in anomalies:
            x = event.index / float(recording.sample_rate)
            ax.axvline(x=x, color="#d62728", linestyle="--", alpha=0.45)
            ax.text(
                x,
                ax.get_ylim()[1],
                event.detector,
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=7,
                color="#d62728",
            )
        ax.set_title("AFR Time-Series with Detected Anomalies")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(filtered, bins=50, color="#2ca02c", alpha=0.7)
        ax2.set_title("Filtered Value Distribution")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    return output
