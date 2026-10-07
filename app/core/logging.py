"""Structured logging built on structlog.

* JSON lines in production, coloured console output in development.
* ``request_id`` (and ``user_id`` when known) are bound per request through
  contextvars so every log line of a request can be correlated.
* The last few hundred records are kept in :class:`LogBuffer`, which feeds the
  admin "live ingestion stream" over WebSocket and ``GET /admin/logs/recent``.
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from datetime import UTC, datetime
from typing import Any

import structlog

LOG_BUFFER_SIZE = 500


class LogBuffer:
    """Thread-safe-enough ring buffer of recent structured log records."""

    def __init__(self, maxlen: int = LOG_BUFFER_SIZE) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._listeners: list[Any] = []

    def append(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:  # pragma: no cover - listeners must never break logging
                pass

    def recent(self, limit: int = 100, *, min_level: str | None = None) -> list[dict[str, Any]]:
        records = list(self._records)
        if min_level:
            threshold = logging.getLevelName(min_level.upper())
            if isinstance(threshold, int):
                records = [
                    r
                    for r in records
                    if isinstance(logging.getLevelName(str(r.get("level", "info")).upper()), int)
                    and logging.getLevelName(str(r.get("level", "info")).upper()) >= threshold
                ]
        return records[-limit:]

    def add_listener(self, callback: Any) -> None:
        self._listeners.append(callback)

    def remove_listener(self, callback: Any) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)


log_buffer = LogBuffer()


def _buffer_processor(
    logger: logging.Logger, method_name: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    record = {
        "timestamp": event_dict.get("timestamp") or datetime.now(UTC).isoformat(),
        "level": event_dict.get("level", method_name),
        "logger": event_dict.get("logger", ""),
        "event": event_dict.get("event", ""),
    }
    for key in ("request_id", "source", "symbol", "job", "user_id"):
        if key in event_dict:
            record[key] = event_dict[key]
    log_buffer.append(record)
    return event_dict


def configure_logging(level: str = "INFO", *, as_json: bool = False) -> None:
    """Configure structlog and route the stdlib loggers (uvicorn, pymongo...) through it."""
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        _buffer_processor,
    ]

    renderer: Any
    if as_json:
        renderer = structlog.processors.JSONRenderer()
        shared_processors.append(structlog.processors.format_exc_info)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("pymongo", "httpx", "httpcore", "websockets", "urllib3", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(uvicorn_logger).handlers.clear()
        logging.getLogger(uvicorn_logger).propagate = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


__all__ = ["LogBuffer", "configure_logging", "get_logger", "log_buffer"]
