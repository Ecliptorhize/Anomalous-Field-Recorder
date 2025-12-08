"""Virtual sensor for testing pipelines without hardware."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Deque, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _now_ts() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SimulatedSensor:
    """Simple sensor simulator producing noise plus occasional anomalies."""

    sample_rate: float = 200.0
    baseline: float = 0.0
    noise: float = 0.2
    anomaly_magnitude: float = 4.0
    anomaly_chance: float = 0.02
    seed: int | None = None
    history: Deque[Tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=10_000))

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def set_sample_rate(self, sample_rate: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = float(sample_rate)

    def stream(
        self,
        duration_s: float = 5.0,
        *,
        realtime: bool = False,
        callback: Callable[[datetime, float, bool], None] | None = None,
    ) -> Iterable[Tuple[datetime, float, bool]]:
        total_samples = int(duration_s * self.sample_rate)
        step = 1.0 / self.sample_rate
        start = _now_ts()
        for idx in range(total_samples):
            ts = start + timedelta(seconds=idx * step)
            # numpy timedelta trick keeps the timestamp monotonic without drift
            value = float(self.baseline + self.rng.normal(scale=self.noise))
            is_anomaly = self.rng.random() < self.anomaly_chance
            if is_anomaly:
                value += float(self.anomaly_magnitude * self.rng.choice([-1, 1]))
            self.history.append((ts, value))
            if callback:
                callback(ts, value, is_anomaly)
            yield ts, value, is_anomaly
            if realtime:
                time.sleep(step)

    def generate(self, duration_s: float = 5.0) -> Tuple[List[datetime], List[float], List[int]]:
        timestamps: List[datetime] = []
        values: List[float] = []
        flags: List[int] = []
        for ts, value, is_anomaly in self.stream(duration_s=duration_s, realtime=False):
            timestamps.append(ts)
            values.append(value)
            flags.append(1 if is_anomaly else 0)
        return timestamps, values, flags

    def visualize(self, duration_s: float = 5.0, output_path: str | None = None) -> str:
        """Generate a quick-look plot showing injected anomalies."""

        import matplotlib.pyplot as plt  # imported lazily to avoid heavy startup

        timestamps, values, flags = self.generate(duration_s=duration_s)
        anomalies = [(t, v) for t, v, f in zip(timestamps, values, flags) if f]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(timestamps, values, label="signal", color="#1f77b4")
        if anomalies:
            ax.scatter(
                [t for t, _ in anomalies],
                [v for _, v in anomalies],
                color="#d62728",
                label="anomaly",
                zorder=3,
            )
        ax.set_title("Simulated Sensor Stream")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)

        output = output_path or "simulated_stream.png"
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        return output

    def to_dataframe(self, duration_s: float = 5.0) -> pd.DataFrame:
        timestamps, values, flags = self.generate(duration_s=duration_s)
        return pd.DataFrame({"timestamp": timestamps, "value": values, "anomaly": flags})
