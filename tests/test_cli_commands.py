from __future__ import annotations

import json
from pathlib import Path

from anomalous_field_recorder import cli


def test_cli_analyze_supports_filters_and_output(tmp_path: Path) -> None:
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps([0.0, 1.0, 0.0, -1.0, 0.5, -0.5]), encoding="utf-8")
    output_path = tmp_path / "analysis.json"

    cli.main(
        [
            "analyze",
            str(samples_path),
            "--sample-rate",
            "50.0",
            "--band",
            "1.0",
            "20.0",
            "--notch",
            "10.0",
            "--anomaly-threshold",
            "2.0",
            "--output",
            str(output_path),
            "--json",
        ]
    )

    analysis = json.loads(output_path.read_text(encoding="utf-8"))
    assert analysis["filters"]["band"] == [1.0, 20.0]
    assert analysis["filters"]["notch"] == 10.0
    assert "bandpower" in analysis and "spectral_entropy" in analysis
    assert analysis["anomalies"]["threshold"] == 2.0


def test_cli_synth_generates_multichannel_payload(tmp_path: Path) -> None:
    output_path = tmp_path / "synth.json"

    cli.main(
        [
            "synth",
            "--channels",
            "2",
            "--duration-s",
            "1.2",
            "--sample-rate",
            "100.0",
            "--component",
            "5.0",
            "0.5",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "channels" in payload and len(payload["channels"]) == 2
    assert payload["events_s"]
    assert payload["sample_rate"] == 100.0


def test_cli_normalize_writes_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "domain: clinical_lab",
                "sample_type: serum",
                "analyte: cbc",
                "lab_id: lab42",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "normalized.json"

    cli.main(["normalize", str(config_path), "--output", str(output_path), "--include-messages"])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["normalized"]["panel"] == "unspecified"
    assert payload["domain"] == "clinical_lab"
    assert payload["errors"] == []
