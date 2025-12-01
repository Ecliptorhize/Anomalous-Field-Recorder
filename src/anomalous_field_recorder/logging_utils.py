"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping


def configure_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    """Configure global logging. Respects AFR_JSON_LOGS env override."""

    if json_logs is None:
        json_logs = os.getenv("AFR_JSON_LOGS", "false").lower() == "true"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s" if json_logs else "%(levelname)s:%(name)s:%(message)s",
    )


def log_event(logger: logging.Logger, event: str, *, json_logs: bool | None = None, **fields: Any) -> None:
    """Emit a structured log event."""

    if json_logs is None:
        json_logs = os.getenv("AFR_JSON_LOGS", "false").lower() == "true"

    payload = {"event": event, **fields}
    if json_logs:
        logger.info(json.dumps(payload))
    else:
        logger.info(payload)
