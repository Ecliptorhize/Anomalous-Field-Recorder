"""FastAPI service exposing acquire/process/report operations."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException

from .config import validate_config
from .pipeline import process_dataset, simulate_acquisition
from .registry import list_runs, record_run
from .reporting import generate_report
from .signals import (
    apply_filters,
    compute_bandpower,
    compute_signal_metrics,
    compute_spectral_entropy,
    compute_spectral_metrics,
    generate_multichannel_eeg,
    generate_synthetic_series,
    score_anomalies,
)

try:
    __version__ = metadata.version("anomalous-field-recorder")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.4.0"


def create_app(registry_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Anomalous Field Recorder API", version=__version__)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> Dict[str, str]:
        return {"version": __version__}

    @app.post("/validate")
    def validate(config: Dict[str, Any]) -> Dict[str, Any]:
        return validate_config(config).as_dict()

    @app.post("/normalize")
    def normalize(config: Dict[str, Any] = Body(...), include_messages: bool = False) -> Dict[str, Any]:
        result = validate_config(config)
        if include_messages:
            return {
                "normalized": result.normalized,
                "errors": result.errors,
                "warnings": result.warnings,
                "domain": result.domain,
            }
        return result.normalized

    @app.post("/acquire")
    def acquire(config_path: str, output_dir: str) -> Dict[str, Any]:
        try:
            metadata = simulate_acquisition(config_path, output_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if registry_path:
            record_run(registry_path, "acquire", str(output_dir), metadata.get("status", "unknown"), metadata.get("domain_profile", {}).get("domain", "unknown"))
        return metadata

    @app.post("/process")
    def process(raw_dir: str, processed_dir: str) -> Dict[str, Any]:
        summary = process_dataset(raw_dir, processed_dir)
        if registry_path:
            record_run(registry_path, "process", str(processed_dir), summary.get("status", "unknown"), summary.get("domain", "unknown"))
        return summary

    @app.post("/report")
    def report(processed_dir: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            path = generate_report(processed_dir, output_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if registry_path:
            record_run(registry_path, "report", str(path), "generated", "unknown")
        return {"report": str(path)}

    @app.post("/analyze")
    def analyze(
        samples: List[float] = Body(...),
        sample_rate: float = Body(1000.0),
        band: Optional[List[float]] = Body(None),
        notch: Optional[float] = Body(None),
        anomaly_threshold: float = Body(3.5),
    ) -> Dict[str, Any]:
        band_tuple = tuple(band) if band else None
        filtered = apply_filters(samples, sample_rate, band=band_tuple, notch=notch) if (band_tuple or notch) else samples
        metrics = compute_signal_metrics(filtered)
        spectral = compute_spectral_metrics(filtered, sample_rate)
        anomalies = score_anomalies(filtered, z_threshold=anomaly_threshold)
        bandpower = compute_bandpower(filtered, sample_rate)
        entropy = compute_spectral_entropy(filtered, sample_rate)
        return {
            "metrics": metrics,
            "spectral": spectral,
            "bandpower": bandpower,
            "spectral_entropy": entropy,
            "anomalies": anomalies,
            "filters": {"band": band, "notch": notch},
        }

    @app.post("/synth")
    def synth(
        duration_s: float = Body(1.0),
        sample_rate: float = Body(1000.0),
        components: Optional[List[Dict[str, float]]] = Body(None),
        noise_std: float = Body(0.02),
        channels: int = Body(0),
    ) -> Dict[str, Any]:
        if channels < 0:
            raise HTTPException(status_code=400, detail="channels cannot be negative")

        if channels:
            channel_data, events = generate_multichannel_eeg(
                num_channels=channels,
                duration_s=duration_s,
                sample_rate=sample_rate,
                base_components=components,
                noise_std=noise_std,
            )
            return {
                "channels": channel_data,
                "events_s": events,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
            }

        samples = generate_synthetic_series(
            duration_s=duration_s,
            sample_rate=sample_rate,
            components=components,
            noise_std=noise_std,
        )
        return {"samples": samples, "sample_rate": sample_rate, "duration_s": duration_s}

    @app.get("/runs")
    def runs(limit: int = 50) -> Dict[str, Any]:
        if not registry_path:
            raise HTTPException(status_code=400, detail="Registry path not configured")
        return {"runs": [r.__dict__ for r in list_runs(registry_path, limit=limit)]}

    return app
