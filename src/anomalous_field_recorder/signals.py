"""Signal generation and analysis helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import signal


def generate_synthetic_series(
    duration_s: float = 2.0,
    sample_rate: float = 1000.0,
    components: Sequence[Mapping[str, float]] | None = None,
    noise_std: float = 0.02,
) -> List[float]:
    """Generate a synthetic time series composed of sinusoids and noise."""

    if components is None:
        components = [{"freq": 5.0, "amplitude": 1.0}, {"freq": 50.0, "amplitude": 0.2}]

    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    series = np.zeros_like(t)
    for comp in components:
        freq = float(comp.get("freq", 0.0))
        amp = float(comp.get("amplitude", 0.0))
        phase = float(comp.get("phase", 0.0))
        series += amp * np.sin(2 * math.pi * freq * t + phase)

    if noise_std:
        series += np.random.normal(scale=noise_std, size=series.shape)

    return series.tolist()


def compute_signal_metrics(samples: Sequence[float]) -> Mapping[str, float]:
    """Compute basic descriptive statistics for a signal."""

    if not samples:
        return {"mean": 0.0, "std": 0.0, "rms": 0.0, "min": 0.0, "max": 0.0}

    arr = np.asarray(samples, dtype=float)
    rms = float(np.sqrt(np.mean(arr**2)))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "rms": rms,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_spectral_metrics(samples: Sequence[float], sample_rate: float) -> Mapping[str, float]:
    """Compute spectral centroid and dominant frequency using FFT."""

    if not samples:
        return {"dominant_freq": 0.0, "spectral_centroid": 0.0}

    arr = np.asarray(samples, dtype=float)
    freqs = np.fft.rfftfreq(arr.size, 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(arr))
    if spectrum.size == 0:
        return {"dominant_freq": 0.0, "spectral_centroid": 0.0}

    dominant_idx = int(np.argmax(spectrum))
    dominant_freq = float(freqs[dominant_idx])
    total_power = np.sum(spectrum)
    if total_power == 0:
        spectral_centroid = 0.0
    else:
        spectral_centroid = float(np.sum(freqs * spectrum) / total_power)

    return {"dominant_freq": dominant_freq, "spectral_centroid": spectral_centroid}


def compute_bandpower(
    samples: Sequence[float],
    sample_rate: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
) -> Mapping[str, float]:
    """Compute bandpower across named frequency bands using Welch PSD."""

    if not samples or sample_rate <= 0:
        return {}

    if bands is None:
        bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 80),
        }

    arr = np.asarray(samples, dtype=float)
    freqs, psd = signal.welch(arr, fs=sample_rate, nperseg=min(256, len(arr)))
    bandpower: dict[str, float] = {}

    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        bandpower[name] = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    total_power = float(np.trapz(psd, freqs)) if freqs.size else 0.0
    if total_power > 0:
        for name, power in list(bandpower.items()):
            bandpower[f"{name}_rel"] = power / total_power

    return bandpower


def apply_filters(
    samples: Sequence[float],
    sample_rate: float,
    band: tuple[float, float] | None = None,
    notch: float | None = None,
) -> List[float]:
    """Apply bandpass and optional notch filtering using Butterworth and IIR filters."""

    if not samples:
        return []

    arr = np.asarray(samples, dtype=float)
    filtered = arr

    if band:
        low, high = band
        nyquist = sample_rate / 2.0
        high = min(high, nyquist - 1e-6)
        low = max(low, 0.001)
        if high <= low:
            return filtered.tolist()
        sos = signal.butter(4, [low, high], btype="band", fs=sample_rate, output="sos")
        filtered = signal.sosfilt(sos, filtered)

    if notch and notch < (sample_rate / 2.0):
        b, a = signal.iirnotch(notch, 30, sample_rate)
        filtered = signal.filtfilt(b, a, filtered)

    return filtered.tolist()


def score_anomalies(samples: Sequence[float], z_threshold: float = 3.5) -> Mapping[str, Any]:
    """Compute a simple z-score based anomaly score."""

    if not samples:
        return {"count": 0, "threshold": z_threshold, "indices": []}

    arr = np.asarray(samples, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr) or 1e-9
    zscores = np.abs((arr - mean) / std)
    indices = np.where(zscores > z_threshold)[0].tolist()

    return {"count": len(indices), "threshold": z_threshold, "indices": indices}


def ingest_samples(path: str | Path, value_column: str | None = None) -> List[float]:
    """Load samples from CSV, JSON list, or JSONL file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")

    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            values = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, (int, float)):
                        values.append(float(obj))
                    elif isinstance(obj, dict) and value_column and value_column in obj:
                        values.append(float(obj[value_column]))
                except json.JSONDecodeError:
                    continue
            return values
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            return [float(x) for x in loaded if isinstance(x, (int, float))]
        raise ValueError("JSON sample file must contain a list of numbers")

    # CSV or other table-like data using pandas
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency error path
        raise ImportError("pandas is required to ingest CSV samples") from exc

    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - optional engine detail
            raise ImportError("pyarrow or fastparquet is required to read Parquet files") from exc
    else:
        df = pd.read_csv(path)
    if value_column is None:
        value_column = df.columns[0]
    return df[value_column].astype(float).tolist()
