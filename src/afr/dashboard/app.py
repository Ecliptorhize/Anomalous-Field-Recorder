"""Lightweight dashboard for recent anomaly events."""

from __future__ import annotations

from fastapi import FastAPI, Response

from ..storage.base import StorageBackend


def create_dashboard_app(storage: StorageBackend) -> FastAPI:
    app = FastAPI(title="AFR Dashboard", version="1.0")

    @app.get("/events")
    def events(limit: int = 50) -> dict:
        return {"events": list(storage.fetch_recent_events(limit))}

    @app.get("/")
    def index() -> Response:  # pragma: no cover - presentation only
        html = """
        <html>
        <head><title>AFR Dashboard</title></head>
        <body>
        <h1>Anomalous Field Recorder Dashboard</h1>
        <p>Recent events are available at <a href="/events">/events</a>.</p>
        </body>
        </html>
        """
        return Response(content=html, media_type="text/html")

    return app
