"""Time helpers used across ingestion, candles and forecasting."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_INTERVAL_RE = re.compile(r"^(\d+)([smhdw])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

SUPPORTED_INTERVALS = ("1m", "5m", "15m", "1h", "4h", "1d")


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def interval_seconds(interval: str) -> int:
    match = _INTERVAL_RE.match(interval.strip().lower())
    if not match:
        raise ValueError(f"Unsupported interval '{interval}'")
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def interval_delta(interval: str) -> timedelta:
    return timedelta(seconds=interval_seconds(interval))


def floor_time(value: datetime, interval: str) -> datetime:
    """Round ``value`` down to the start of its ``interval`` bucket (UTC)."""
    value = ensure_utc(value)
    seconds = interval_seconds(interval)
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)


def parse_range(value: str) -> timedelta:
    """Parse a human range such as ``30d``, ``12h`` or ``4w``."""
    return interval_delta(value)


def from_ms(ms: int | float) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def to_ms(value: datetime) -> int:
    return int(ensure_utc(value).timestamp() * 1000)


__all__ = [
    "SUPPORTED_INTERVALS",
    "ensure_utc",
    "floor_time",
    "from_ms",
    "interval_delta",
    "interval_seconds",
    "parse_range",
    "to_ms",
    "utcnow",
]
