"""SQLite-backed storage backend."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .base import StorageBackend


class SQLiteBackend(StorageBackend):
    """Lightweight SQLite storage for windows and anomaly events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                sample_rate REAL NOT NULL,
                values_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detector TEXT NOT NULL,
                score REAL NOT NULL,
                threshold REAL,
                started_at TEXT NOT NULL,
                metadata_json TEXT
            )
            """
        )
        self.conn.commit()

    def store_window(self, window: Sequence[float] | np.ndarray, sample_rate: float, started_at: datetime) -> None:
        payload = json.dumps(list(map(float, window)))
        self.conn.execute(
            "INSERT INTO windows (started_at, sample_rate, values_json) VALUES (?, ?, ?)",
            (started_at.isoformat(), float(sample_rate), payload),
        )
        self.conn.commit()

    def store_event(
        self,
        detector: str,
        score: float,
        threshold: float | None,
        started_at: datetime,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload = json.dumps(metadata or {})
        self.conn.execute(
            "INSERT INTO events (detector, score, threshold, started_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
            (detector, float(score), threshold, started_at.isoformat(), payload),
        )
        self.conn.commit()

    def fetch_recent_events(self, limit: int = 50) -> Iterable[Mapping[str, object]]:
        rows = self.conn.execute(
            "SELECT detector, score, threshold, started_at, metadata_json FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        for detector, score, threshold, started_at, metadata_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else {}
            yield {
                "detector": detector,
                "score": score,
                "threshold": threshold,
                "started_at": started_at,
                "metadata": metadata,
            }

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
