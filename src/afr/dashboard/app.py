"""Dashboard exposing live charts, anomaly flags, and run history."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response, WebSocket

from ..storage.base import StorageBackend


class DashboardState:
    """In-memory state store for live dashboard elements."""

    def __init__(self, storage: StorageBackend | None = None, max_points: int = 2_000) -> None:
        self.storage = storage
        self.points: Deque[dict] = deque(maxlen=max_points)
        self.channels: Dict[str, Deque[dict]] = {}
        self.spectrogram: Deque[dict] = deque(maxlen=200)
        self.events: Deque[dict] = deque(maxlen=200)
        self.runs: Deque[dict] = deque(maxlen=100)

    def add_point(
        self,
        timestamp: str,
        value: float,
        *,
        is_anomaly: bool = False,
        detector: str | None = None,
        channel: str = "default",
    ) -> None:
        self.points.append({"timestamp": timestamp, "value": value, "channel": channel})
        if channel not in self.channels:
            maxlen = self.points.maxlen or 2_000
            self.channels[channel] = deque(maxlen=maxlen)
        self.channels[channel].append({"timestamp": timestamp, "value": value})
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

    def add_spectrogram(self, timestamp: str, freqs: List[float], magnitudes: List[float], channel: str = "default") -> None:
        self.spectrogram.appendleft({"timestamp": timestamp, "freqs": freqs, "magnitudes": magnitudes, "channel": channel})

    def add_run(self, name: str, status: str, path: str) -> None:
        self.runs.appendleft({"name": name, "status": status, "path": path})

    def snapshot(self, limit: int = 500) -> Dict[str, Any]:
        pts = list(self.points)[-limit:]
        return {
            "points": pts,
            "channels": {k: list(v)[-limit:] for k, v in self.channels.items()},
            "spectrogram": list(self.spectrogram)[:limit],
            "events": list(self.events)[:limit],
            "runs": list(self.runs),
        }


def _token_dependency(api_token: Optional[str]):
    async def verifier(x_api_key: str | None = Header(default=None)) -> None:
        if api_token and x_api_key != api_token:
            raise HTTPException(status_code=401, detail="invalid token")

    return verifier


def create_dashboard_app(
    storage: StorageBackend | None = None,
    dataset_dir: str | Path = "datasets",
    *,
    api_token: str | None = None,
) -> FastAPI:
    state = DashboardState(storage=storage)
    app = FastAPI(title="AFR Dashboard", version="1.1")
    require_token = _token_dependency(api_token)

    @app.get("/health")
    def health(token=Depends(require_token)) -> dict[str, str]:  # type: ignore[unused-argument]
        return {"status": "ok", "points": str(len(state.points)), "channels": list(state.channels)}

    @app.get("/datasets")
    def datasets(token=Depends(require_token)) -> dict[str, List[str]]:  # type: ignore[unused-argument]
        root = Path(dataset_dir)
        files = [str(p) for p in root.glob("*.csv")]
        return {"datasets": files}

    @app.get("/runs")
    def runs(token=Depends(require_token)) -> dict[str, List[dict]]:  # type: ignore[unused-argument]
        return {"runs": list(state.runs)}

    @app.get("/events")
    def events(limit: int = 50, token=Depends(require_token)) -> dict:  # type: ignore[unused-argument]
        if storage:
            try:
                return {"events": list(storage.fetch_recent_events(limit))}
            except Exception:  # pragma: no cover
                pass
        return {"events": list(state.events)[:limit]}

    @app.get("/snapshot")
    def snapshot(limit: int = 500, token=Depends(require_token)) -> dict:  # type: ignore[unused-argument]
        return state.snapshot(limit=limit)

    @app.get("/metrics")
    def metrics(token=Depends(require_token)) -> Mapping[str, Any]:  # type: ignore[unused-argument]
        return {
            "points": len(state.points),
            "events": len(state.events),
            "runs": len(state.runs),
            "channels": {k: len(v) for k, v in state.channels.items()},
        }

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket) -> None:  # pragma: no cover - runtime path
        await ws.accept()
        await ws.send_json(state.snapshot(limit=300))
        while True:
            await asyncio.sleep(1.0)
            await ws.send_json(
                {
                    "points": list(state.points)[-100:],
                    "channels": {k: list(v)[-300:] for k, v in state.channels.items()},
                    "spectrogram": list(state.spectrogram)[:20],
                    "events": list(state.events)[:50],
                }
            )

    @app.get("/")
    def index() -> Response:  # pragma: no cover - presentation only
        html = """
        <html>
        <head>
            <title>AFR Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
            <style>
                body { font-family: 'Inter', system-ui, -apple-system, sans-serif; background: linear-gradient(120deg, #0c1220, #0b1f3a); color: #e4e9f3; margin: 0; padding: 24px;}
                h1 { margin: 0 0 12px 0; font-weight: 700; }
                .grid { display: grid; gap: 16px; grid-template-columns: 2fr 1fr; }
                .card { background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.25);}
                #chart, #spectrogram { width: 100%; height: 320px; }
                #events div { padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.08);}
                .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; background: rgba(255,255,255,0.08); margin-right: 4px; font-size: 12px;}
            </style>
        </head>
        <body>
        <h1>AFR Live Dashboard</h1>
        <div class="grid">
          <div class="card">
            <h3>Signals</h3>
            <div id="chart"></div>
          </div>
          <div class="card">
            <h3>Spectrogram</h3>
            <div id="spectrogram"></div>
          </div>
        </div>
        <div class="card" style="margin-top:16px;">
            <h3>Anomalies</h3>
            <div id="events"></div>
        </div>
        <script>
            const chart = echarts.init(document.getElementById('chart'));
            const specChart = echarts.init(document.getElementById('spectrogram'));
            const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/stream');
            const option = {
                tooltip: { trigger: 'axis' },
                legend: { textStyle: { color: '#e4e9f3' }},
                xAxis: { type: 'category', data: [], axisLabel: { color: '#a7b3d1'} },
                yAxis: { type: 'value', axisLabel: { color: '#a7b3d1'} },
                series: []
            };
            const specOption = {
                tooltip: {},
                xAxis: { type: 'value', name: 'Freq (Hz)', axisLabel: { color: '#a7b3d1'} },
                yAxis: { type: 'value', name: 'Power', axisLabel: { color: '#a7b3d1'} },
                series: [{ type: 'heatmap', data: [], blurSize: 12, pointSize: 14 }]
            };
            chart.setOption(option);
            specChart.setOption(specOption);
            ws.onmessage = (event) => {
                const payload = JSON.parse(event.data);
                if (payload.channels) {
                    option.series = [];
                    Object.keys(payload.channels).forEach((ch) => {
                        const data = payload.channels[ch];
                        option.series.push({ name: ch, type: 'line', data: data.map(p => p.value), smooth: true });
                        option.xAxis.data = data.map(p => p.timestamp);
                    });
                    chart.setOption(option, true);
                }
                if (payload.spectrogram) {
                    const heat = payload.spectrogram.flatMap((row, idx) => row.freqs.map((f, i) => [f, idx, row.magnitudes[i]]));
                    specOption.series[0].data = heat;
                    specChart.setOption(specOption, true);
                }
                if (payload.events) {
                    document.getElementById('events').innerHTML = payload.events.map(e => `<div><span class="pill">${e.detector || 'detector'}</span>${e.timestamp}</div>`).join('');
                }
            };
        </script>
        </body>
        </html>
        """
        return Response(content=html, media_type="text/html")

    app.state.dashboard = state
    return app
