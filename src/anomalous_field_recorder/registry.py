"""Lightweight SQLite registry for runs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class RunRecord:
    id: int
    kind: str
    location: str
    status: str
    domain: str
    created_at: str


def init_registry(db_path: str | Path) -> Path:
    """Ensure registry database and table exist."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                location TEXT NOT NULL,
                status TEXT NOT NULL,
                domain TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def record_run(db_path: str | Path, kind: str, location: str, status: str, domain: str) -> int:
    """Insert a run record."""

    db_path = init_registry(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO runs (kind, location, status, domain, created_at) VALUES (?, ?, ?, ?, ?)",
            (kind, location, status, domain, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_runs(db_path: str | Path, limit: Optional[int] = 50) -> List[RunRecord]:
    """Fetch recent runs."""

    db_path = init_registry(db_path)
    conn = sqlite3.connect(db_path)
    try:
        sql = "SELECT id, kind, location, status, domain, created_at FROM runs ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            rows = conn.execute(sql, (limit,)).fetchall()
        else:
            rows = conn.execute(sql).fetchall()
        return [RunRecord(*row) for row in rows]
    finally:
        conn.close()


def purge_runs(db_path: str | Path) -> None:
    """Delete all run records."""

    conn = sqlite3.connect(init_registry(db_path))
    try:
        conn.execute("DELETE FROM runs")
        conn.commit()
    finally:
        conn.close()

