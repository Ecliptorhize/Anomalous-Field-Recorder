"""TimescaleDB backend for higher-throughput storage."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, Mapping, Sequence

import numpy as np

from .base import StorageBackend

try:  # pragma: no cover - dependency optional
    import psycopg2
except ModuleNotFoundError:  # pragma: no cover - fallback
    psycopg2 = None


class TimescaleBackend(StorageBackend):
    """TimescaleDB-backed storage backend.

    If psycopg2 is unavailable, the backend degrades to an in-memory buffer so
    that the interface remains usable in constrained environments.
    """

    def __init__(self, dsn: str, create_hypertables: bool = True) -> None:
        self.dsn = dsn
        self.create_hypertables = create_hypertables
        if psycopg2:
            try:
                self.conn = psycopg2.connect(dsn)  # type: ignore[arg-type]
            except Exception:
                self.conn = None
        else:
            self.conn = None
        self._events: list[Mapping[str, object]] = []
        self._windows: list[Mapping[str, object]] = []
        if self.conn:
            self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS afr_windows (
                started_at TIMESTAMPTZ NOT NULL,
                sample_rate DOUBLE PRECISION NOT NULL,
                values_json JSONB NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS afr_events (
                started_at TIMESTAMPTZ NOT NULL,
                detector TEXT NOT NULL,
                score DOUBLE PRECISION NOT NULL,
                threshold DOUBLE PRECISION,
                metadata_json JSONB
            );
            """
        )
        if self.create_hypertables:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            cur.execute("SELECT create_hypertable('afr_windows', 'started_at', if_not_exists => TRUE);")
            cur.execute("SELECT create_hypertable('afr_events', 'started_at', if_not_exists => TRUE);")
        self.conn.commit()

    def store_window(self, window: Sequence[float] | np.ndarray, sample_rate: float, started_at: datetime) -> None:
        payload = json.dumps(list(map(float, window)))
        if self.conn:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO afr_windows (started_at, sample_rate, values_json) VALUES (%s, %s, %s)",
                (started_at, float(sample_rate), payload),
            )
            self.conn.commit()
        else:
            self._windows.append({"started_at": started_at, "sample_rate": float(sample_rate), "values": window})

    def store_event(
        self,
        detector: str,
        score: float,
        threshold: float | None,
        started_at: datetime,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload = json.dumps(metadata or {})
        if self.conn:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO afr_events (started_at, detector, score, threshold, metadata_json) VALUES (%s, %s, %s, %s, %s)",
                (started_at, detector, float(score), threshold, payload),
            )
            self.conn.commit()
        else:
            self._events.append(
                {
                    "started_at": started_at.isoformat(),
                    "detector": detector,
                    "score": float(score),
                    "threshold": threshold,
                    "metadata": metadata or {},
                }
            )

    def fetch_recent_events(self, limit: int = 50) -> Iterable[Mapping[str, object]]:
        if self.conn:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT detector, score, threshold, started_at, metadata_json FROM afr_events ORDER BY started_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            for detector, score, threshold, started_at, metadata_json in rows:
                yield {
                    "detector": detector,
                    "score": score,
                    "threshold": threshold,
                    "started_at": started_at.isoformat(),
                    "metadata": metadata_json or {},
                }
        else:
            yield from self._events[:limit]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
