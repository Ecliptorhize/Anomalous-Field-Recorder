"""Streaming source adapters for live ingestion."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class StreamingSource:
    """Iterable source contract."""

    def __iter__(self) -> Iterator[Sequence[float]]:  # pragma: no cover - interface only
        raise NotImplementedError


class SimulatedSource(StreamingSource):
    """Default simulated source used for demos and tests."""

    def __init__(self, batch_size: int = 128, anomaly_chance: float = 0.1, anomaly_scale: float = 2.0) -> None:
        self.batch_size = batch_size
        self.anomaly_chance = anomaly_chance
        self.anomaly_scale = anomaly_scale
        self._rng = np.random.default_rng()

    def __iter__(self) -> Iterator[Sequence[float]]:
        while True:
            base = self._rng.normal(scale=0.1, size=self.batch_size)
            if self._rng.random() < self.anomaly_chance:
                base = base + self._rng.normal(scale=self.anomaly_scale, size=self.batch_size)
            yield base.tolist()


class CSVTailSource(StreamingSource):
    """Tails a CSV file and yields numeric batches."""

    def __init__(self, path: str | Path, batch_size: int = 128, poll_interval: float = 0.5) -> None:
        self.path = Path(path)
        self.batch_size = batch_size
        self.poll_interval = poll_interval

    def __iter__(self) -> Iterator[Sequence[float]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open("r", encoding="utf-8") as handle:
            # skip header if present
            header = handle.readline()
            try:
                float(header.split(",")[0])
                handle.seek(0)
            except Exception:
                pass

            buffer: list[float] = []
            while True:
                pos = handle.tell()
                line = handle.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    handle.seek(pos)
                    continue
                try:
                    value = float(line.split(",")[0])
                    buffer.append(value)
                except ValueError:
                    continue
                if len(buffer) >= self.batch_size:
                    yield buffer[: self.batch_size]
                    buffer = buffer[self.batch_size :]


class MQTTSource(StreamingSource):
    """MQTT ingestion source (requires paho-mqtt)."""

    def __init__(self, topic: str, host: str = "localhost", port: int = 1883, batch_size: int = 128) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install paho-mqtt to use MQTTSource") from exc

        self.batch_size = batch_size
        self._buffer: list[float] = []
        self._queue: list[list[float]] = []
        self._client = mqtt.Client()
        self._client.on_message = self._on_message
        self._client.connect(host, port, keepalive=30)
        self._client.subscribe(topic)
        self._client.loop_start()

    def _on_message(self, _client, _userdata, msg) -> None:  # pragma: no cover - network path
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            if isinstance(data, list):
                values = [float(x) for x in data]
            else:
                values = [float(data)]
            self._buffer.extend(values)
            while len(self._buffer) >= self.batch_size:
                batch = self._buffer[: self.batch_size]
                self._buffer = self._buffer[self.batch_size :]
                self._queue.append(batch)
        except Exception:
            logger.exception("MQTTSource failed to parse message")

    def __iter__(self) -> Iterator[Sequence[float]]:
        while True:
            if self._queue:
                yield self._queue.pop(0)
            else:
                time.sleep(0.05)


class WebSocketSource(StreamingSource):
    """WebSocket ingestion source (requires websockets)."""

    def __init__(self, url: str, batch_size: int = 128) -> None:
        try:
            import websockets  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install websockets to use WebSocketSource") from exc
        self.url = url
        self.batch_size = batch_size
        self._queue: list[list[float]] = []
        self._buffer: list[float] = []
        self._ws_module = websockets
        self._start()

    def _start(self) -> None:
        import asyncio

        async def _consume() -> None:  # pragma: no cover - network path
            async with self._ws_module.connect(self.url) as ws:
                async for message in ws:
                    try:
                        data = json.loads(message)
                        values = data if isinstance(data, list) else [float(data)]
                        self._buffer.extend(float(x) for x in values)
                        while len(self._buffer) >= self.batch_size:
                            batch = self._buffer[: self.batch_size]
                            self._buffer = self._buffer[self.batch_size :]
                            self._queue.append(batch)
                    except Exception:
                        logger.exception("WebSocketSource failed to parse message")

        loop = asyncio.get_event_loop()
        if not loop.is_running():
            loop.create_task(_consume())
        else:
            # fire-and-forget in background thread
            import threading

            threading.Thread(target=lambda: asyncio.run(_consume()), daemon=True).start()

    def __iter__(self) -> Iterator[Sequence[float]]:
        while True:
            if self._queue:
                yield self._queue.pop(0)
            else:
                time.sleep(0.05)


def build_source(cfg: Mapping[str, object] | None) -> Callable[[], Iterable[Sequence[float]]]:
    """Factory for streaming sources based on config mapping."""

    if cfg is None:
        return lambda: SimulatedSource()

    if callable(cfg):
        return cfg  # type: ignore[return-value]

    source_type = str(cfg.get("type", "simulated")).lower()  # type: ignore[union-attr]
    if source_type in {"simulated", "demo"}:
        batch_size = int(cfg.get("batch_size", 128))  # type: ignore[arg-type]
        return lambda: SimulatedSource(batch_size=batch_size)
    if source_type in {"csv", "tail"}:
        path = cfg.get("path")
        if not path:
            raise ValueError("CSV source requires 'path'")
        batch_size = int(cfg.get("batch_size", 128))  # type: ignore[arg-type]
        poll_interval = float(cfg.get("poll_interval", 0.5))  # type: ignore[arg-type]
        return lambda: CSVTailSource(path, batch_size=batch_size, poll_interval=poll_interval)
    if source_type == "mqtt":
        topic = cfg.get("topic")
        if not topic:
            raise ValueError("MQTT source requires 'topic'")
        host = cfg.get("host", "localhost")
        port = int(cfg.get("port", 1883))  # type: ignore[arg-type]
        batch_size = int(cfg.get("batch_size", 128))  # type: ignore[arg-type]
        return lambda: MQTTSource(topic=str(topic), host=str(host), port=port, batch_size=batch_size)
    if source_type == "websocket":
        url = cfg.get("url")
        if not url:
            raise ValueError("WebSocket source requires 'url'")
        batch_size = int(cfg.get("batch_size", 128))  # type: ignore[arg-type]
        return lambda: WebSocketSource(url=str(url), batch_size=batch_size)

    raise ValueError(f"Unknown streaming source type '{source_type}'")
