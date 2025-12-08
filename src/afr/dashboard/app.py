"""Dashboard exposing live charts, anomaly flags, and run history."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List

from fastapi import FastAPI, Response, WebSocket

from ..storage.base import StorageBackend


class DashboardState:
    """In-memory state store for live dashboard elements."""

    def __init__(self, storage: StorageBackend | None = None, max_points: int = 2_000) -> None:
        self.storage = storage
        self.points: Deque[dict] = deque(maxlen=max_points)
        self.events: Deque[dict] = deque(maxlen=200)
        self.runs: Deque[dict] = deque(maxlen=100)

    def add_point(self, timestamp: str, value: float, is_anomaly: bool = False, detector: str | None = None) -> None:
        self.points.append({"timestamp": timestamp, "value": value})
        if is_anomaly:
            event = {"timestamp": timestamp, "value": value, "detector": detector or "unknown"}
            self.events.appendleft(event)
            if self.storage:
                try:
                    self.storage.store_event(
                        detector=detector or "dashboard",
                        score=value,
                        threshold=None,
                        started_at=datetime.now(timezone.utc),
                        metadata={},
                    )
                except Exception:  # pragma: no cover - defensive logging path
                    pass

    def add_run(self, name: str, status: str, path: str) -> None:
        self.runs.appendleft({"name": name, "status": status, "path": path})

    def snapshot(self, limit: int = 500) -> Dict[str, Any]:
        pts = list(self.points)[-limit:]
        return {
            "points": pts,
            "events": list(self.events)[:limit],
            "runs": list(self.runs),
        }


def create_dashboard_app(storage: StorageBackend | None = None, dataset_dir: str | Path = "datasets") -> FastAPI:
    state = DashboardState(storage=storage)
    app = FastAPI(title="AFR Dashboard", version="1.1")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "points": str(len(state.points))}

    @app.get("/datasets")
    def datasets() -> dict[str, List[str]]:
        root = Path(dataset_dir)
        files = [str(p) for p in root.glob("*.csv")]
        return {"datasets": files}

    @app.get("/runs")
    def runs() -> dict[str, List[dict]]:
        return {"runs": list(state.runs)}

    @app.get("/events")
    def events(limit: int = 50) -> dict:
        if storage:
            try:
                return {"events": list(storage.fetch_recent_events(limit))}
            except Exception:  # pragma: no cover
                pass
        return {"events": list(state.events)[:limit]}

    @app.get("/snapshot")
    def snapshot(limit: int = 500) -> dict:
        return state.snapshot(limit=limit)

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:  # pragma: no cover - runtime path
        await ws.accept()
        await ws.send_json(state.snapshot(limit=300))
        while True:
            await asyncio.sleep(1.0)
            await ws.send_json({"points": list(state.points)[-100:], "events": list(state.events)[:50]})

    @app.get("/")
    def index() -> Response:  # pragma: no cover - presentation only
        html = """
        <html>
        <head>
            <title>AFR Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        </head>
        <body style="font-family: Arial, sans-serif;">
        <h1>Anomalous Field Recorder Dashboard</h1>
        <div id="chart" style="width: 100%; height: 400px;"></div>
        <div id="events"></div>
        <script>
            const chart = echarts.init(document.getElementById('chart'));
            const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/stream');
            const option = {
                title: {text: 'Live Stream'},
                xAxis: {type: 'category', data: []},
                yAxis: {type: 'value'},
                series: [{name: 'value', type: 'line', data: []}]
            };
            chart.setOption(option);
            ws.onmessage = (event) => {
                const payload = JSON.parse(event.data);
                if (payload.points) {
                    option.xAxis.data = payload.points.map(p => p.timestamp);
                    option.series[0].data = payload.points.map(p => p.value);
                    chart.setOption(option);
                }
                if (payload.events) {
                    document.getElementById('events').innerHTML = '<h3>Anomalies</h3>' + payload.events.map(e => `<div>${e.timestamp} :: ${e.detector || ''}</div>`).join('');
                }
            };
        </script>
        </body>
        </html>
        """
        return Response(content=html, media_type="text/html")

    app.state.dashboard = state
    return app
