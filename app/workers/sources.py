"""Tracks the health of every ingestion source (the admin Data Ingestion Hub rows)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.timeutils import utcnow
from app.db.ingestion_logs import append_ingestion_log
from app.modules.market.models import IngestionSource, SourceStatus
from app.modules.market.repository import IngestionSourceRepository

logger = structlog.get_logger(__name__)

PERSIST_INTERVAL = 5.0


@dataclass
class SourceState:
    name: str
    label: str
    kind: str
    engine: str
    description: str = ""
    status: SourceStatus = SourceStatus.STOPPED
    enabled: bool = True
    started_at: datetime | None = None
    last_event_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    events_total: int = 0
    error_count: int = 0
    restarts: int = 0
    latency_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    event_times: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    healthy_seconds: float = 0.0
    tracked_seconds: float = 0.0
    _last_sample: float = field(default_factory=time.monotonic)

    def events_per_minute(self) -> float:
        cutoff = time.monotonic() - 60
        return float(sum(1 for t in self.event_times if t >= cutoff))

    def sample_uptime(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_sample
        self._last_sample = now
        if not self.enabled:
            return
        self.tracked_seconds += elapsed
        if self.status in (SourceStatus.CONNECTED, SourceStatus.ACTIVE):
            self.healthy_seconds += elapsed

    def uptime_pct(self) -> float | None:
        if self.tracked_seconds <= 0:
            return None
        return round(100 * self.healthy_seconds / self.tracked_seconds, 2)


class SourceTracker:
    def __init__(self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.repo = IngestionSourceRepository(db)
        self.sources: dict[str, SourceState] = {}
        self._task: asyncio.Task[None] | None = None

    # ---------------------------------------------------------- registry
    def register(
        self, name: str, *, label: str, kind: str, engine: str, description: str = "", enabled: bool = True
    ) -> SourceState:
        state = self.sources.get(name)
        if state is None:
            state = SourceState(
                name=name, label=label, kind=kind, engine=engine, description=description, enabled=enabled
            )
            self.sources[name] = state
        return state

    def enabled(self, name: str) -> bool:
        state = self.sources.get(name)
        return state.enabled if state else False

    # ------------------------------------------------------------ events
    def set_status(self, name: str, status: SourceStatus, *, message: str | None = None) -> None:
        state = self.sources[name]
        changed = state.status != status
        state.status = status
        if status in (SourceStatus.CONNECTED, SourceStatus.ACTIVE) and state.started_at is None:
            state.started_at = utcnow()
        if changed:
            logger.info("source_status", source=name, status=str(status))
            asyncio.get_running_loop().create_task(self.log(name, message or f"{state.label}: {status}", level="info"))

    def event(self, name: str, *, count: int = 1, latency_ms: float | None = None, **extra: Any) -> None:
        state = self.sources[name]
        state.events_total += count
        now = time.monotonic()
        for _ in range(min(count, 100)):
            state.event_times.append(now)
        state.last_event_at = utcnow()
        if latency_ms is not None:
            state.latency_ms = round(latency_ms, 1)
        if extra:
            state.extra.update(extra)
        if state.status not in (SourceStatus.CONNECTED, SourceStatus.ACTIVE):
            state.status = SourceStatus.ACTIVE if state.kind != "websocket" else SourceStatus.CONNECTED

    def error(self, name: str, message: str) -> None:
        state = self.sources[name]
        state.error_count += 1
        state.last_error = message[:500]
        state.last_error_at = utcnow()
        state.status = SourceStatus.ERROR
        logger.warning("source_error", source=name, error=message[:200])
        asyncio.get_running_loop().create_task(self.log(name, message[:300], level="error"))

    def restarted(self, name: str) -> None:
        self.sources[name].restarts += 1

    async def log(self, name: str, message: str, *, level: str = "info", **fields: Any) -> None:
        try:
            await append_ingestion_log(self.db, name, message, level=level, **fields)
        except Exception as exc:  # pragma: no cover - logging must never break ingestion
            logger.debug("ingestion_log_failed", error=str(exc))

    # --------------------------------------------------------- persistence
    async def start(self) -> None:
        # Pick up enabled/disabled flags from a previous run and persist initial rows.
        for state in self.sources.values():
            existing = await self.repo.get_by_name(state.name)
            if existing is not None:
                state.enabled = existing.enabled
            state.status = SourceStatus.STOPPED if state.enabled else SourceStatus.DISABLED
        await self.persist()
        self._task = asyncio.create_task(self._persist_loop(), name="source-tracker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for state in self.sources.values():
            if state.status in (SourceStatus.CONNECTED, SourceStatus.ACTIVE):
                state.status = SourceStatus.STOPPED
        await self.persist()

    async def _persist_loop(self) -> None:
        while True:
            await asyncio.sleep(PERSIST_INTERVAL)
            try:
                await self.refresh_flags()
                await self.persist()
            except Exception as exc:  # pragma: no cover
                logger.warning("source_persist_failed", error=str(exc))

    async def refresh_flags(self) -> None:
        """Admins toggle ``enabled`` through the API; honour it on the next cycle."""
        docs = await self.repo.find({"name": {"$in": list(self.sources)}}, projection={"name": 1, "enabled": 1})
        for doc in docs:
            state = self.sources.get(doc.name)
            if state and state.enabled != doc.enabled:
                state.enabled = doc.enabled
                state.status = SourceStatus.DISABLED if not doc.enabled else SourceStatus.STOPPED
                await self.log(state.name, f"{state.label} {'enabled' if doc.enabled else 'disabled'} by admin")

    async def persist(self) -> None:
        for state in self.sources.values():
            state.sample_uptime()
            await self.repo.upsert_source(
                state.name,
                label=state.label,
                kind=state.kind,
                engine=state.engine,
                description=state.description,
                status=str(state.status),
                started_at=state.started_at,
                heartbeat_at=utcnow(),
                last_event_at=state.last_event_at,
                last_error=state.last_error,
                last_error_at=state.last_error_at,
                events_total=state.events_total,
                events_per_minute=state.events_per_minute(),
                latency_ms=state.latency_ms,
                uptime_pct=state.uptime_pct(),
                error_count=state.error_count,
                restarts=state.restarts,
                extra=state.extra,
            )

    def snapshot(self) -> list[IngestionSource]:
        return [
            IngestionSource(
                name=s.name,
                label=s.label,
                kind=s.kind,
                engine=s.engine,
                description=s.description,
                status=s.status,
                enabled=s.enabled,
                started_at=s.started_at,
                last_event_at=s.last_event_at,
                last_error=s.last_error,
                last_error_at=s.last_error_at,
                events_total=s.events_total,
                events_per_minute=s.events_per_minute(),
                latency_ms=s.latency_ms,
                uptime_pct=s.uptime_pct(),
                error_count=s.error_count,
                restarts=s.restarts,
                extra=s.extra,
            )
            for s in self.sources.values()
        ]


__all__ = ["SourceState", "SourceTracker"]
