from __future__ import annotations

"""Data profiling helpers for quick dataset inspection."""

import json
from datetime import timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api import types as pd_types


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load various tabular formats into a DataFrame."""

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                if raw and isinstance(raw[0], dict):
                    return pd.DataFrame(raw)
                if raw and all(isinstance(x, (int, float)) for x in raw):
                    return pd.DataFrame({"value": raw})
                return pd.DataFrame({"value": raw})
            if isinstance(raw, dict):
                return pd.DataFrame([raw])
            raise ValueError("Unsupported JSON structure for profiling")
        # default to CSV/TSV and let pandas infer delimiters
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset {path} is empty") from exc


def _to_iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _detect_anomaly_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if str(c).lower() in {"anomaly", "is_anomaly", "flag", "label"}]
    for col in candidates:
        series = df[col]
        if pd_types.is_bool_dtype(series) or (
            pd_types.is_numeric_dtype(series) and series.dropna().isin([0, 1]).all()
        ):
            return str(col)

    for col in df.columns:
        series = df[col]
        if pd_types.is_bool_dtype(series):
            return str(col)
        if pd_types.is_numeric_dtype(series):
            uniq = series.dropna().unique()
            if len(uniq) <= 2 and set(np.unique(uniq)).issubset({0, 1}):
                return str(col)
    return None


def profile_dataset(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    value_columns: Sequence[str] | None = None,
    sample_rate: float | None = None,
    preview_rows: int = 5,
) -> dict[str, Any]:
    """Profile a dataset for quick quality and metadata checks."""

    df = _load_dataframe(Path(path))
    if df.empty:
        raise ValueError(f"Dataset {path} is empty")

    numeric_cols = [str(col) for col in df.columns if pd_types.is_numeric_dtype(df[col])]
    if value_columns:
        selected = [col for col in value_columns if col in df.columns]
        numeric_cols = [str(col) for col in selected if pd_types.is_numeric_dtype(df[col])]
        if not numeric_cols and selected:
            raise ValueError("None of the requested value columns are numeric")

    missing_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}
    zero_variance = [col for col in numeric_cols if df[col].dropna().nunique() <= 1]

    stats: dict[str, Any] = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        stats[col] = {
            "mean": float(series.mean()) if not series.empty else 0.0,
            "std": float(series.std(ddof=0)) if not series.empty else 0.0,
            "min": float(series.min()) if not series.empty else 0.0,
            "max": float(series.max()) if not series.empty else 0.0,
        }

    anomaly_col = _detect_anomaly_column(df)
    flagged = 0
    if anomaly_col:
        series = df[anomaly_col]
        if pd_types.is_bool_dtype(series):
            flagged = int(series.fillna(False).sum())
        elif pd_types.is_numeric_dtype(series):
            flagged = int(pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0).sum())

    sampling = {
        "timestamp_column": timestamp_column if timestamp_column in df.columns else None,
        "start": None,
        "end": None,
        "duration_s": None,
        "inferred_sample_rate_hz": float(sample_rate) if sample_rate else None,
        "max_gap_s": None,
        "duplicate_timestamps": 0,
    }
    if timestamp_column in df.columns:
        ts = pd.to_datetime(df[timestamp_column], errors="coerce").dropna().sort_values()
        if not ts.empty:
            sampling["start"] = _to_iso(ts.iloc[0])
            sampling["end"] = _to_iso(ts.iloc[-1])
            sampling["duration_s"] = float((ts.iloc[-1] - ts.iloc[0]).total_seconds())
            deltas = ts.diff().dt.total_seconds().dropna()
            if sampling["inferred_sample_rate_hz"] is None and not deltas.empty:
                median_delta = deltas.median()
                if median_delta > 0:
                    sampling["inferred_sample_rate_hz"] = float(1.0 / median_delta)
            if not deltas.empty:
                sampling["max_gap_s"] = float(deltas.max())
            sampling["duplicate_timestamps"] = int(ts.duplicated().sum())

    preview = json.loads(df.head(preview_rows).to_json(orient="records", date_format="iso"))

    rows = int(len(df))
    return {
        "path": str(Path(path).resolve()),
        "rows": rows,
        "columns": [str(col) for col in df.columns],
        "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
        "numeric_columns": numeric_cols,
        "missing_counts": missing_counts,
        "zero_variance_columns": zero_variance,
        "stats": stats,
        "sampling": sampling,
        "anomalies": {
            "flag_column": anomaly_col,
            "flagged": flagged,
            "ratio": (flagged / rows) if rows else 0.0,
        },
        "preview": preview,
    }
