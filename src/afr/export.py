"""Packaging helpers to bundle AFR runs for sharing."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def _load_result(run_dir: Path) -> Mapping[str, object]:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _coerce_formats(formats: Sequence[str] | None) -> list[str]:
    if not formats:
        return ["csv"]
    return [f.lower() for f in formats]


def export_run(run_dir: str | Path, output_path: str | Path, formats: Sequence[str] | None = None) -> Path:
    """Package a processed run directory into a portable archive.

    Includes result.json, reports, and optional dataset conversions (CSV/Parquet/HDF5).
    """

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    formats = _coerce_formats(formats)

    result = _load_result(run_dir)
    report_paths = [
        run_dir / Path(result.get("report", {}).get("output_json", "result.json")).name,
        run_dir / Path(result.get("report", {}).get("output_markdown", "report.md")).name,
        run_dir / Path(result.get("report", {}).get("output_pdf", "report.pdf")).name,
    ]
    files_to_add = [p for p in report_paths if p.exists()]

    dataset_path = Path(result.get("metadata", {}).get("source", "")) if isinstance(result, Mapping) else None
    converted: list[Path] = []

    if dataset_path and dataset_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            if "csv" in formats:
                converted.append(dataset_path)
            if "parquet" in formats:
                parquet_path = run_dir / "dataset.parquet"
                df.to_parquet(parquet_path)  # type: ignore[call-arg]
                converted.append(parquet_path)
            if "hdf5" in formats or "hdf" in formats:
                hdf_path = run_dir / "dataset.h5"
                df.to_hdf(hdf_path, key="dataset", mode="w")
                converted.append(hdf_path)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to package dataset: %s", exc)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(dataset_path) if dataset_path else None,
        "report_files": [p.name for p in files_to_add],
        "conversions": [p.name for p in converted],
        "metadata": result.get("metadata", {}),
        "detectors": result.get("detectors", []),
        "filters": result.get("filters", []),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files_to_add.append(manifest_path)

    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_add + converted:
            zf.write(file_path, arcname=file_path.name)

    # clean up temp conversions if needed
    for path in converted:
        if path.parent == run_dir:
            try:
                path.unlink()
            except OSError:
                pass

    return output_path
