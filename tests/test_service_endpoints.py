from __future__ import annotations

from fastapi.testclient import TestClient

from anomalous_field_recorder import create_app


def test_service_analyze_endpoint_returns_metrics() -> None:
    app = create_app()
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        json={
            "samples": [0.0, 1.0, 0.0, -1.0, 0.5],
            "sample_rate": 50.0,
            "band": [1.0, 20.0],
            "notch": 10.0,
            "anomaly_threshold": 2.0,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data and "spectral_entropy" in data and "bandpower" in data
    assert data["filters"]["band"] == [1.0, 20.0]
    assert data["filters"]["notch"] == 10.0
    assert data["anomalies"]["threshold"] == 2.0


def test_service_synth_endpoint_generates_channels() -> None:
    app = create_app()
    client = TestClient(app)

    resp = client.post(
        "/synth",
        json={
            "channels": 2,
            "duration_s": 0.5,
            "sample_rate": 64.0,
            "components": [{"freq": 5.0, "amplitude": 0.5}],
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert "channels" in payload and len(payload["channels"]) == 2
    assert payload["sample_rate"] == 64.0
    assert payload["events_s"]


def test_service_normalize_endpoint_applies_defaults() -> None:
    app = create_app()
    client = TestClient(app)

    config = {
        "domain": "clinical_lab",
        "sample_type": "serum",
        "analyte": "cbc",
        "lab_id": "lab42",
    }
    resp = client.post("/normalize", json=config, params={"include_messages": "true"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["normalized"]["panel"] == "unspecified"
    assert payload["domain"] == "clinical_lab"
    assert payload["errors"] == []
