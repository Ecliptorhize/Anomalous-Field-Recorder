from __future__ import annotations

from pathlib import Path

import pytest

from anomalous_field_recorder import (
    apply_filters,
    compute_bandpower,
    compute_signal_metrics,
    create_app,
    generate_report,
    generate_synthetic_series,
    list_runs,
    load_experiment_config,
    process_dataset,
    record_run,
    simulate_acquisition,
    summarize_domain,
    validate_config,
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


def test_domain_summary_in_pipeline(tmp_path: Path) -> None:
    config_path = tmp_path / "lab.yml"
    config_path.write_text(
        "\n".join(
            [
                "domain: clinical_lab",
                "lab_id: medlab-12",
                "instrument: analyzer-x",
                "sample_type: serum",
                "analyte: sodium",
            ]
        )
    )

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    simulate_acquisition(config_path, raw_dir)
    summary = process_dataset(raw_dir, processed_dir)

    assert summary["domain"] == "clinical_lab"
    assert summary["instrument"] == "analyzer-x"
    assert any("all required metadata present" in flag for flag in summary["quality_flags"])


def test_summarize_domain_highlights_missing_fields() -> None:
    profile = summarize_domain({"modality": "mri", "patient_id": "anon"})
    assert profile["domain"] == "medical_imaging"
    assert "exam_id" in profile["missing_required"]


def test_validate_config_applies_defaults() -> None:
    result = validate_config({"domain": "clinical_lab", "sample_type": "serum", "analyte": "cbc", "lab_id": "lab"})
    assert result.domain == "clinical_lab"
    assert "panel" in result.normalized
    assert not result.errors


def test_signal_metrics_and_filtering() -> None:
    series = generate_synthetic_series(duration_s=0.1, sample_rate=100.0, components=[{"freq": 5.0, "amplitude": 1.0}], noise_std=0.0)
    filtered = apply_filters(series, sample_rate=100.0, band=(1.0, 10.0))
    metrics = compute_signal_metrics(filtered)
    assert metrics["rms"] > 0


def test_bandpower_computation() -> None:
    series = generate_synthetic_series(duration_s=1.0, sample_rate=256.0, components=[{"freq": 10.0, "amplitude": 1.0}], noise_std=0.0)
    bp = compute_bandpower(series, sample_rate=256.0)
    assert bp["alpha"] > 0


def test_registry_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    record_id = record_run(db, "acquire", "path", "ok", "field_engineering")
    runs = list_runs(db)
    assert runs[0].id == record_id


def test_service_health_endpoint(tmp_path: Path) -> None:
    app = create_app(registry_path=tmp_path / "registry.db")
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
