"""Lightweight in-process metrics for the admin health dashboard.

Not a Prometheus replacement: just enough counters and latency samples to show
request volume, error rate, p95 latency and third-party provider usage.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ProviderStats:
    """Usage of one external API (Binance, CoinGecko, Moralis, OpenAI ...)."""

    name: str
    calls: int = 0
    errors: int = 0
    last_call_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    last_latency_ms: float | None = None
    rate_limit_remaining: int | None = None
    rate_limit_reset_at: datetime | None = None
    latencies: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def record(self, latency_ms: float, *, error: str | None = None) -> None:
        self.calls += 1
        self.last_call_at = datetime.now(UTC)
        self.last_latency_ms = latency_ms
        self.latencies.append(latency_ms)
        if error:
            self.errors += 1
            self.last_error = error[:300]
            self.last_error_at = self.last_call_at

    def as_dict(self) -> dict[str, Any]:
        avg = sum(self.latencies) / len(self.latencies) if self.latencies else None
        return {
            "name": self.name,
            "calls": self.calls,
            "errors": self.errors,
            "error_rate": round(self.errors / self.calls, 4) if self.calls else 0.0,
            "avg_latency_ms": round(avg, 1) if avg is not None else None,
            "last_latency_ms": self.last_latency_ms,
            "last_call_at": self.last_call_at,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset_at": self.rate_limit_reset_at,
        }


class MetricsRegistry:
    def __init__(self) -> None:
        self.started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()
        self.requests_total = 0
        self.responses_by_class: Counter[str] = Counter()
        self.requests_by_route: Counter[str] = Counter()
        self.latencies_ms: deque[float] = deque(maxlen=2000)
        self.error_timestamps: deque[float] = deque(maxlen=1000)
        self.providers: dict[str, ProviderStats] = {}
        self.websocket_connections = 0

    # -------------------------------------------------------------- http
    def record_request(self, route: str, status_code: int, latency_ms: float) -> None:
        self.requests_total += 1
        self.responses_by_class[f"{status_code // 100}xx"] += 1
        self.requests_by_route[route] += 1
        self.latencies_ms.append(latency_ms)
        if status_code >= 500:
            self.error_timestamps.append(time.monotonic())

    def errors_in_window(self, window_seconds: float) -> int:
        cutoff = time.monotonic() - window_seconds
        return sum(1 for ts in self.error_timestamps if ts >= cutoff)

    def latency_percentile(self, percentile: float) -> float | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, max(0, round(percentile / 100 * (len(ordered) - 1))))
        return round(ordered[index], 2)

    # --------------------------------------------------------- providers
    def provider(self, name: str) -> ProviderStats:
        stats = self.providers.get(name)
        if stats is None:
            stats = ProviderStats(name=name)
            self.providers[name] = stats
        return stats

    # ----------------------------------------------------------- summary
    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started_monotonic

    def snapshot(self) -> dict[str, Any]:
        total = self.requests_total or 1
        return {
            "started_at": self.started_at,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "requests_total": self.requests_total,
            "responses_by_class": dict(self.responses_by_class),
            "error_rate": round(self.responses_by_class["5xx"] / total, 4),
            "latency_ms": {
                "p50": self.latency_percentile(50),
                "p95": self.latency_percentile(95),
                "p99": self.latency_percentile(99),
            },
            "top_routes": self.requests_by_route.most_common(10),
            "websocket_connections": self.websocket_connections,
            "providers": [p.as_dict() for p in self.providers.values()],
        }


metrics = MetricsRegistry()

__all__ = ["MetricsRegistry", "ProviderStats", "metrics"]
