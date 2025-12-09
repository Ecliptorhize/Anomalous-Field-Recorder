"""Alerting and sensor health helpers for streaming AFR deployments."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib import request

from .anomaly.engine import DetectionResult

logger = logging.getLogger(__name__)


# --- Alert rules and channels -------------------------------------------------


@dataclass
class AlertRule:
    """Rule describing when to emit an alert."""

    name: str
    metric: str
    threshold: float
    comparison: str = "gt"
    hysteresis: float = 0.0
    cooldown_s: float = 10.0
    dedup_key: str | None = None
    severity: str = "warning"

    _last_fired: float = field(default=0.0, init=False, repr=False)
    _last_value: float = field(default=0.0, init=False, repr=False)

    def should_fire(self, value: float) -> bool:
        now = time.time()
        if self.cooldown_s and (now - self._last_fired) < self.cooldown_s:
            return False

        self._last_value = value
        if self.comparison == "gt":
            triggered = value > self.threshold + self.hysteresis
        elif self.comparison == "lt":
            triggered = value < self.threshold - self.hysteresis
        else:
            triggered = False

        if triggered:
            self._last_fired = now
        return triggered


class AlertChannel:
    """Base alert destination."""

    def send(self, payload: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LoggingChannel(AlertChannel):
    """Default alert channel emitting to logger/stdout."""

    def __init__(self, level: int = logging.WARNING) -> None:
        self.level = level

    def send(self, payload: Mapping[str, Any]) -> None:
        logger.log(self.level, "[alert] %s", json.dumps(payload, default=str))


class WebhookChannel(AlertChannel):
    """Sends alerts to an HTTP webhook."""

    def __init__(self, url: str, headers: Optional[Mapping[str, str]] = None, timeout: int = 5) -> None:
        self.url = url
        self.headers = {**({"Content-Type": "application/json"} if headers is None else headers)}
        self.timeout = timeout

    def send(self, payload: Mapping[str, Any]) -> None:  # pragma: no cover - network
        data = json.dumps(payload, default=str).encode("utf-8")
        req = request.Request(self.url, data=data, headers=self.headers)
        with request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            if resp.status >= 300:
                raise RuntimeError(f"Webhook returned {resp.status}")


class SlackWebhookChannel(WebhookChannel):
    """Simple Slack-compatible webhook payload."""

    def send(self, payload: Mapping[str, Any]) -> None:  # pragma: no cover - network
        text = payload.get("text") or payload.get("message") or "AFR alert"
        enriched = {"text": text, "attachments": [{"color": "#ff5252", "fields": [{"title": k, "value": str(v)} for k, v in payload.items()]}]}
        super().send(enriched)


class GPIOChannel(AlertChannel):
    """Stub for hardware-triggered alerts (user supplies a callable)."""

    def __init__(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        self.callback = callback

    def send(self, payload: Mapping[str, Any]) -> None:
        try:
            self.callback(payload)
        except Exception:  # pragma: no cover - defensive
            logger.exception("GPIO alert callback failed")


# --- Health monitoring -------------------------------------------------------


@dataclass
class HealthStatus:
    ok: bool
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class SensorHealthMonitor:
    """Tracks drift and noise floor to flag sensor health issues."""

    def __init__(
        self,
        mean_drift_tol: float = 5.0,
        std_ceiling: float = 5.0,
        min_sample_rate: float | None = None,
    ) -> None:
        self.mean_drift_tol = mean_drift_tol
        self.std_ceiling = std_ceiling
        self.min_sample_rate = min_sample_rate
        self._baseline_mean: float | None = None
        self._baseline_std: float | None = None

    def update_baseline(self, mean: float, std: float) -> None:
        self._baseline_mean = mean
        self._baseline_std = std

    def check(
        self,
        features: Mapping[str, float],
        sample_rate: float,
    ) -> HealthStatus:
        warnings: list[str] = []
        mean = features.get("mean", 0.0)
        std = features.get("std", 0.0)

        if self._baseline_mean is None:
            self.update_baseline(mean, std)

        if self._baseline_mean is not None and abs(mean - self._baseline_mean) > self.mean_drift_tol:
            warnings.append(f"mean drift {mean:.3f} vs baseline {self._baseline_mean:.3f}")

        if self._baseline_std is not None and std > (self._baseline_std + self.std_ceiling):
            warnings.append(f"noise/std high {std:.3f} vs baseline {self._baseline_std:.3f}")

        if self.min_sample_rate and sample_rate < self.min_sample_rate:
            warnings.append(f"sample_rate below expected {sample_rate} < {self.min_sample_rate}")

        return HealthStatus(ok=not warnings, warnings=warnings, metrics={"mean": mean, "std": std})


# --- Alert manager -----------------------------------------------------------


class AlertManager:
    """Evaluates rules and emits alerts over configured channels."""

    def __init__(self, rules: Iterable[AlertRule], channels: Iterable[AlertChannel]) -> None:
        self.rules = list(rules)
        self.channels = list(channels) or [LoggingChannel()]

    def evaluate(
        self,
        *,
        features: Mapping[str, float],
        detections: Sequence[DetectionResult],
        health: HealthStatus | None,
    ) -> None:
        metrics = dict(features)
        for det in detections:
            metrics[f"detection:{det.detector}"] = det.score

        triggered: list[dict[str, Any]] = []
        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                continue
            if rule.should_fire(value):
                payload = {
                    "rule": rule.name,
                    "metric": rule.metric,
                    "value": value,
                    "threshold": rule.threshold,
                    "comparison": rule.comparison,
                    "severity": rule.severity,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "health": health.metrics if health else {},
                    "warnings": health.warnings if health else [],
                }
                triggered.append(payload)

        if triggered:
            self._dispatch(triggered)

    def _dispatch(self, alerts: List[Mapping[str, Any]]) -> None:
        for alert in alerts:
            for channel in self.channels:
                try:
                    channel.send(alert)
                except Exception:  # pragma: no cover - defensive path
                    logger.exception("Alert channel %s failed", channel.__class__.__name__)


def manager_from_config(cfg: Mapping[str, Any]) -> AlertManager:
    """Construct an AlertManager from config mapping."""

    rules_cfg = cfg.get("rules", []) if cfg else []
    channels_cfg = cfg.get("channels", []) if cfg else []

    rules = [
        AlertRule(
            name=rc.get("name", rc.get("metric", "rule")),
            metric=rc.get("metric", "mean"),
            threshold=float(rc.get("threshold", 1.0)),
            comparison=str(rc.get("comparison", "gt")),
            hysteresis=float(rc.get("hysteresis", 0.0)),
            cooldown_s=float(rc.get("cooldown_s", 10.0)),
            severity=str(rc.get("severity", "warning")),
        )
        for rc in rules_cfg
    ]

    channels: list[AlertChannel] = []
    for ch in channels_cfg:
        ctype = str(ch.get("type", "log")).lower()
        if ctype in {"log", "logging"}:
            channels.append(LoggingChannel())
        elif ctype in {"webhook", "http"}:
            url = ch.get("url")
            if not url:
                raise ValueError("Webhook channel requires 'url'")
            channels.append(WebhookChannel(url=str(url), headers=ch.get("headers")))
        elif ctype == "slack":
            url = ch.get("url")
            if not url:
                raise ValueError("Slack channel requires 'url'")
            channels.append(SlackWebhookChannel(url=str(url)))
        else:
            raise ValueError(f"Unknown alert channel type '{ctype}'")

    return AlertManager(rules=rules, channels=channels or [LoggingChannel()])
