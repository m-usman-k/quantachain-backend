"""Worker runtime: owns the scheduler, long-running streams and the source tracker.

Modules contribute jobs through an optional ``app.modules.<name>.jobs`` module
exposing ``register(runtime)``. The runtime can live inside the API process
(``RUN_WORKERS_IN_API=true``, handy in development) or in the dedicated worker
process started with ``python -m app.worker``.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.events import EventBus, event_bus
from app.workers.scheduler import JobSpec, Scheduler
from app.workers.sources import SourceTracker

logger = structlog.get_logger(__name__)

StreamFactory = Callable[[], Awaitable[None]]

JOB_MODULES = (
    "app.modules.market.jobs",
    "app.modules.onchain.jobs",
    "app.modules.sentiment.jobs",
    "app.modules.prediction.jobs",
    "app.modules.fraud.jobs",
    "app.modules.admin.jobs",
    "app.modules.research.jobs",
    "app.modules.trading.jobs",
)


@dataclass
class StreamSpec:
    name: str
    factory: StreamFactory
    module: str = "core"
    enabled: bool = True


class WorkerRuntime:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, bus: EventBus | None = None
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.scheduler = Scheduler(db, self.settings)
        self.tracker = SourceTracker(db, self.settings)
        self.streams: dict[str, StreamSpec] = {}
        self._stream_tasks: dict[str, asyncio.Task[None]] = {}
        self._stopping = asyncio.Event()
        self.started = False

    # ---------------------------------------------------------- registry
    def add_job(self, spec: JobSpec) -> None:
        self.scheduler.add(spec)

    def add_stream(self, name: str, factory: StreamFactory, *, module: str = "core", enabled: bool = True) -> None:
        if name in self.streams:
            raise ValueError(f"Duplicate stream {name}")
        self.streams[name] = StreamSpec(name=name, factory=factory, module=module, enabled=enabled)

    def discover(self) -> list[str]:
        registered = []
        for path in JOB_MODULES:
            try:
                module = import_module(path)
            except ModuleNotFoundError as exc:
                if exc.name and path.startswith(exc.name):
                    continue
                raise
            register = getattr(module, "register", None)
            if register is None:
                continue
            register(self)
            registered.append(path)
        return registered

    # --------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self.started:
            return
        self._stopping.clear()
        modules = self.discover()
        await self.tracker.start()
        await self.scheduler.start()
        for spec in self.streams.values():
            if spec.enabled:
                self._stream_tasks[spec.name] = asyncio.create_task(self._supervise(spec), name=f"stream:{spec.name}")
        self.started = True
        logger.info(
            "worker_runtime_started", modules=modules, jobs=len(self.scheduler.jobs), streams=len(self._stream_tasks)
        )

    async def stop(self) -> None:
        if not self.started:
            return
        self._stopping.set()
        for task in self._stream_tasks.values():
            task.cancel()
        for task in self._stream_tasks.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._stream_tasks.clear()
        await self.scheduler.stop()
        await self.tracker.stop()
        self.started = False
        logger.info("worker_runtime_stopped")

    async def _supervise(self, spec: StreamSpec) -> None:
        """Restart a stream whenever it exits or crashes, with exponential backoff."""
        backoff = 1.0
        while not self._stopping.is_set():
            try:
                await spec.factory()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "stream_crashed", stream=spec.name, error=f"{type(exc).__name__}: {exc}", retry_in=round(backoff, 1)
                )
                if spec.name in self.tracker.sources:
                    self.tracker.error(spec.name, f"{type(exc).__name__}: {exc}")
            if self._stopping.is_set():
                return
            if spec.name in self.tracker.sources:
                self.tracker.restarted(spec.name)
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60.0)

    # ------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "jobs": {
                name: {
                    "module": spec.module,
                    "interval_seconds": spec.interval_seconds,
                    **self.scheduler._snapshot(name),
                }
                for name, spec in self.scheduler.jobs.items()
            },
            "streams": {
                name: {
                    "module": s.module,
                    "enabled": s.enabled,
                    "running": name in self._stream_tasks and not self._stream_tasks[name].done(),
                }
                for name, s in self.streams.items()
            },
            "sources": [s.model_dump() for s in self.tracker.snapshot()],
        }


__all__ = ["JOB_MODULES", "StreamSpec", "WorkerRuntime"]
