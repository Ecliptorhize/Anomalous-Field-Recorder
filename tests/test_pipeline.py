from __future__ import annotations

from pathlib import Path

import pytest

from anomalous_field_recorder import (
    generate_report,
    load_experiment_config,
    process_dataset,
    simulate_acquisition,
)


def test_load_experiment_config_supports_simple_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("""site: test-station\nsample_rate: 1000\n""")

    config = load_experiment_config(config_path)

    assert config == {"site": "test-station", "sample_rate": 1000}


def test_pipeline_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("site: integration-test\n")

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    metadata = simulate_acquisition(config_path, raw_dir)
    assert metadata["status"] == "acquired"
    assert metadata["config"]["site"] == "integration-test"

    summary = process_dataset(raw_dir, processed_dir)
    assert summary["source"].endswith(str(raw_dir))
    assert summary["config_keys"] == ["site"]

    report_path = generate_report(processed_dir)
    assert report_path.exists()
    report_text = report_path.read_text()
    assert "integration-test" in report_text or "site" in report_text


def test_generate_report_requires_summary(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        generate_report(processed_dir)
