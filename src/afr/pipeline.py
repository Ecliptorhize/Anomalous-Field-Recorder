"""End-to-end processing pipeline: load → preprocess → filter → detect → report."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .detection import MADDetector, RollingMeanVarianceDetector, ZScoreDetector
from .detection.machine_learning import AutoencoderDetector, IsolationForestDetector, OneClassSVMDetector
from .anomaly.detectors.adaptive_threshold import AdaptiveThresholdDetector
from .anomaly.detectors.matrix_profile import MatrixProfileDetector
from .anomaly.detectors.spectral_kurtosis import SpectralKurtosisDetector
from .filters import FilterChain
from .models import AnomalyEvent, Recording, RecordingMetadata
from .reporting import render_markdown_report, write_pdf_report

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configurable parameters for the offline pipeline."""

    sample_rate: float | None = None
    window_size: int = 256
    stride: int = 128
    filters: list[Mapping[str, object]] = field(default_factory=list)
    detectors: list[Mapping[str, object]] = field(
        default_factory=lambda: [
            {"type": "zscore", "threshold": 3.5, "window": 256},
            {"type": "mad", "threshold": 3.5},
            {"type": "rolling_mean_variance", "threshold": 3.0, "window": 128},
        ]
    )
    report_title: str = "AFR Analysis Report"
    output_json: str = "result.json"
    output_pdf: str = "report.pdf"
    output_markdown: str = "report.md"

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object]) -> "PipelineConfig":
        kwargs = {k: v for k, v in cfg.items() if hasattr(cls, k)}  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        raw = Path(path)
        if not raw.exists():
            raise FileNotFoundError(raw)
        text = raw.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text) if raw.suffix.lower() in {".yml", ".yaml"} else json.loads(text)
        if not isinstance(cfg, Mapping):
            raise ValueError("Pipeline config file must contain a mapping/object at the top level")
        return cls.from_mapping(cfg)


def load_dataset(dataset_path: str | Path, sample_rate: float | None = None) -> Recording:
    """Load a CSV dataset into a Recording model."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset {path} is empty")

    if "value" in df.columns:
        value_column = "value"
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            raise ValueError("Dataset must contain at least one numeric column")
        value_column = numeric_cols[0]
    values = df[value_column].astype(float).tolist()

    timestamps: list[pd.Timestamp] = []
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna().tolist()

    if sample_rate is None and timestamps:
        deltas = pd.Series(timestamps).diff().dt.total_seconds().dropna()
        median_delta = deltas.median() if not deltas.empty else None
        if median_delta and median_delta > 0:
            sample_rate = 1.0 / median_delta
    sample_rate = float(sample_rate or 100.0)

    sensor_type = None
    if "sensor" in df.columns:
        sensor_type = str(df["sensor"].iloc[0])
    elif len(df.columns) > 1 and df.columns[1] != value_column:
        sensor_type = df.columns[1]

    metadata = RecordingMetadata(
        source=str(path),
        sample_rate=sample_rate,
        sensor_type=sensor_type or "unknown",
        tags=["pipeline"],
    )

    ts_parsed = None
    if timestamps:
        converted = []
        for t in timestamps:
            py_dt = t.to_pydatetime()
            if py_dt.tzinfo is None:
                py_dt = py_dt.replace(tzinfo=timezone.utc)
            else:
                py_dt = py_dt.astimezone(timezone.utc)
            converted.append(py_dt)
        ts_parsed = converted

    return Recording(
        id=path.stem,
        sample_rate=sample_rate,
        values=values,
        timestamps=ts_parsed,
        metadata=metadata,
    )


def _build_detectors(config: Iterable[Mapping[str, object]]) -> list[Any]:
    detectors: list[Any] = []
    for det_cfg in config:
        det_type = str(det_cfg.get("type") or det_cfg.get("name") or "").lower()  # type: ignore[union-attr]
        params = {k: v for k, v in det_cfg.items() if k not in {"type", "name"}}  # type: ignore[union-attr]
        try:
            if det_type in {"zscore", "z-score"}:
                detectors.append(ZScoreDetector(**params))
            elif det_type in {"rolling_mean_variance", "rolling"}:
                detectors.append(RollingMeanVarianceDetector(**params))
            elif det_type in {"mad", "median_absolute_deviation"}:
                detectors.append(MADDetector(**params))
            elif det_type in {"adaptive_threshold", "adaptive"}:
                detectors.append(AdaptiveThresholdDetector(**params))
            elif det_type in {"matrix_profile", "discord"}:
                detectors.append(MatrixProfileDetector(**params))
            elif det_type in {"spectral_kurtosis", "kurtosis"}:
                detectors.append(SpectralKurtosisDetector(**params))
            elif det_type in {"iforest", "isolation_forest"}:
                detectors.append(IsolationForestDetector(**params))
            elif det_type in {"ocsvm", "one_class_svm", "one-class-svm"}:
                detectors.append(OneClassSVMDetector(**params))
            elif det_type == "autoencoder":
                detectors.append(AutoencoderDetector(**params))
            else:
                raise ValueError(f"Unknown detector type '{det_type}'")
        except ImportError as exc:  # pragma: no cover - optional dependencies
            logger.warning("Skipping detector '%s': %s", det_type, exc)
    return detectors


def _preprocess(values: Sequence[float]) -> np.ndarray:
    """Basic preprocessing: drop NaNs and interpolate gaps."""

    series = pd.Series(values, dtype=float)
    if series.isna().any():
        series = series.interpolate(limit_direction="both")
        series = series.fillna(method="bfill").fillna(method="ffill")
    return series.to_numpy(dtype=float)


def _window_generator(arr: np.ndarray, window_size: int, stride: int) -> Iterable[tuple[np.ndarray, int]]:
    length = arr.size
    for start in range(0, length, stride):
        end = min(start + window_size, length)
        window = arr[start:end]
        if window.size < max(4, window_size // 4):
            break
        yield window, start


def detect_anomalies(
    recording: Recording,
    filtered: np.ndarray,
    detectors: Sequence[Any],
    *,
    window_size: int,
    stride: int,
) -> list[AnomalyEvent]:
    anomalies: list[AnomalyEvent] = []
    for window, start in _window_generator(filtered, window_size, stride):
        for det in detectors:
            score = float(det.score(window))
            if det.predict(window, score=score):
                center_idx = start + (window.size // 2)
                anomalies.append(
                    AnomalyEvent(
                        index=center_idx,
                        timestamp=recording.timestamp_for_index(center_idx),
                        detector=det.name,
                        score=score,
                        threshold=det.threshold,
                        context={"window_start": start, "window_size": int(window.size)},
                    )
                )
    return anomalies


def run_pipeline(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig | Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Execute the full processing pipeline and emit JSON + PDF reports."""

    cfg = config if isinstance(config, PipelineConfig) else PipelineConfig.from_mapping(config or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recording = load_dataset(dataset_path, sample_rate=cfg.sample_rate)
    raw = _preprocess(recording.values)

    filter_chain = FilterChain.from_config(cfg.filters, sample_rate=recording.sample_rate)
    filtered = filter_chain.apply(raw, recording.sample_rate)

    detectors = _build_detectors(cfg.detectors)
    anomalies = detect_anomalies(
        recording,
        filtered,
        detectors,
        window_size=cfg.window_size,
        stride=cfg.stride,
    )

    result = {
        "metadata": recording.metadata.model_dump() if recording.metadata else {},
        "sample_rate": recording.sample_rate,
        "n_samples": len(recording.values),
        "duration_s": recording.duration_seconds(),
        "filters": [f.describe() for f in filter_chain.filters],
        "detectors": [d.describe() if hasattr(d, "describe") else {"name": getattr(d, "name", "detector")} for d in detectors],
        "anomalies": [a.model_dump() for a in anomalies],
        "report": {
            "title": cfg.report_title,
            "output_json": str(output_dir / cfg.output_json),
            "output_pdf": str(output_dir / cfg.output_pdf),
            "output_markdown": str(output_dir / cfg.output_markdown),
        },
    }

    json_output = output_dir / cfg.output_json
    json_output.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")

    markdown_output = output_dir / cfg.output_markdown
    context = {
        "title": cfg.report_title,
        "recording": recording,
        "filters": result["filters"],
        "detectors": result["detectors"],
        "anomalies": anomalies,
        "summary": {
            "n_samples": result["n_samples"],
            "duration_s": result["duration_s"],
            "n_anomalies": len(anomalies),
        },
    }
    render_markdown_report(context, markdown_output)

    pdf_output = output_dir / cfg.output_pdf
    write_pdf_report(recording, raw, filtered, anomalies, pdf_output)

    logger.info(
        "Pipeline complete | samples=%s | anomalies=%s | json=%s",
        result["n_samples"],
        len(anomalies),
        json_output,
    )
    return result
