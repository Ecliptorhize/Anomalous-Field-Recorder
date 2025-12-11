from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anomalous_field_recorder import profile_dataset


def test_profile_dataset_reports_stats_and_sampling(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1s"),
            "value": [0, 1, 2, 3, 4],
            "anomaly": [0, 1, 0, 0, 0],
            "constant": [5, 5, 5, 5, 5],
        }
    )
    df.to_csv(path, index=False)

    profile = profile_dataset(path)

    assert profile["rows"] == 5
    assert "value" in profile["numeric_columns"]
    assert profile["anomalies"]["flagged"] == 1
    assert "constant" in profile["zero_variance_columns"]
    assert profile["sampling"]["inferred_sample_rate_hz"] == pytest.approx(1.0)
    assert profile["sampling"]["duration_s"] == pytest.approx(4.0)
    assert profile["stats"]["value"]["max"] == 4.0
    assert profile["preview"]
