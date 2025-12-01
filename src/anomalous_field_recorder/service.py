"""FastAPI service exposing acquire/process/report operations."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from .config import validate_config
from .pipeline import process_dataset, simulate_acquisition
from .registry import list_runs, record_run
from .reporting import generate_report

try:
    __version__ = metadata.version("anomalous-field-recorder")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.3.0"


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

    @app.get("/runs")
    def runs(limit: int = 50) -> Dict[str, Any]:
        if not registry_path:
            raise HTTPException(status_code=400, detail="Registry path not configured")
        return {"runs": [r.__dict__ for r in list_runs(registry_path, limit=limit)]}

    return app
