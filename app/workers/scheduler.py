"""Asyncio job scheduler with persisted state and admin controls.

Each job runs in its own task loop. State (last run, errors, duration) is
mirrored to the ``scheduler_jobs`` collection so the API process - which may
not host the worker - can display it and request pauses or immediate runs by
flipping flags on the document.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.timeutils import utcnow
from app.db.collections import Collections

logger = structlog.get_logger(__name__)

JobFunc = Callable[[], Awaitable[Any]]
CONTROL_POLL_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 900.0


@dataclass
class JobSpec:
    name: str
    interval_seconds: float
    func: JobFunc
    module: str = "core"
    description: str = ""
    run_on_start: bool = True
    jitter_seconds: float = 0.0
    enabled: bool = True
    timeout_seconds: float | None = None


@dataclass
class JobState:
    running: bool = False
    paused: bool = False
    last_run_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    last_duration_ms: float | None = None
    next_run_at: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_result: Any = None
    run_requested: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    def __init__(self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.col = db[Collections.SCHEDULER_JOBS]
        self.jobs: dict[str, JobSpec] = {}
        self.states: dict[str, JobState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._wakeups: dict[str, asyncio.Event] = {}
        self._stopping = asyncio.Event()
        self.host = f"{socket.gethostname()}:{os.getpid()}"

    # ---------------------------------------------------------- registry
    def add(self, spec: JobSpec) -> None:
        if spec.name in self.jobs:
            raise ValueError(f"Duplicate job name {spec.name}")
        self.jobs[spec.name] = spec
        self.states[spec.name] = JobState()
        self._wakeups[spec.name] = asyncio.Event()

    # --------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self._stopping.clear()
        for name, spec in self.jobs.items():
            await self._persist(
                name,
                {
                    "enabled": spec.enabled,
                    "interval_seconds": spec.interval_seconds,
                    "module": spec.module,
                    "description": spec.description,
                },
            )
            if spec.enabled:
                self._tasks[name] = asyncio.create_task(self._loop(spec), name=f"job:{name}")
        logger.info("scheduler_started", jobs=len(self._tasks))

    async def stop(self) -> None:
        self._stopping.set()
        for event in self._wakeups.values():
            event.set()
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for name in self.jobs:
            await self._persist(name, {"running": False})
        logger.info("scheduler_stopped")

    def request_run(self, name: str) -> None:
        if name in self._wakeups:
            self.states[name].run_requested = True
            self._wakeups[name].set()

    # -------------------------------------------------------------- loop
    async def _loop(self, spec: JobSpec) -> None:
        state = self.states[spec.name]
        if spec.jitter_seconds:
            await asyncio.sleep(random.random() * spec.jitter_seconds)
        if not spec.run_on_start:
            state.next_run_at = utcnow() + timedelta(seconds=spec.interval_seconds)
            await self._wait(spec, spec.interval_seconds)
        while not self._stopping.is_set():
            await self._refresh_controls(spec.name)
            if state.paused and not state.run_requested:
                state.next_run_at = None
                await self._persist(spec.name, self._snapshot(spec.name))
                await self._wait(spec, CONTROL_POLL_SECONDS)
                continue
            state.run_requested = False
            delay = await self._execute(spec)
            state.next_run_at = utcnow() + timedelta(seconds=delay)
            await self._persist(spec.name, self._snapshot(spec.name))
            await self._wait(spec, delay)

    async def _execute(self, spec: JobSpec) -> float:
        state = self.states[spec.name]
        state.running = True
        state.last_run_at = utcnow()
        await self._persist(spec.name, {"running": True, "last_run_at": state.last_run_at})
        started = time.perf_counter()
        log = logger.bind(job=spec.name)
        try:
            if spec.timeout_seconds:
                result = await asyncio.wait_for(spec.func(), spec.timeout_seconds)
            else:
                result = await spec.func()
            state.last_duration_ms = round((time.perf_counter() - started) * 1000, 1)
            state.last_success_at = utcnow()
            state.consecutive_errors = 0
            state.last_result = _jsonable(result)
            log.debug("job_completed", duration_ms=state.last_duration_ms)
            delay = spec.interval_seconds
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.last_duration_ms = round((time.perf_counter() - started) * 1000, 1)
            state.error_count += 1
            state.consecutive_errors += 1
            state.last_error = f"{type(exc).__name__}: {exc}"[:500]
            state.last_error_at = utcnow()
            delay = min(spec.interval_seconds * (2 ** min(state.consecutive_errors, 6)), MAX_BACKOFF_SECONDS)
            log.warning(
                "job_failed", error=state.last_error, retry_in=round(delay, 1), consecutive=state.consecutive_errors
            )
        finally:
            state.running = False
            state.run_count += 1
        return delay

    async def _wait(self, spec: JobSpec, seconds: float) -> None:
        """Sleep up to ``seconds`` but wake early on stop or an admin-requested run."""
        event = self._wakeups[spec.name]
        event.clear()
        deadline = time.monotonic() + seconds
        while not self._stopping.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(event.wait(), min(remaining, CONTROL_POLL_SECONDS))
                if self.states[spec.name].run_requested:
                    return
                event.clear()
            except TimeoutError:
                pass
            # Pick up pause / run-now flags set through the admin API.
            await self._refresh_controls(spec.name)
            if self.states[spec.name].run_requested:
                return

    # ----------------------------------------------------------- control
    async def _refresh_controls(self, name: str) -> None:
        try:
            doc = await self.col.find_one({"name": name}, {"paused": 1, "run_requested_at": 1, "last_run_at": 1})
        except Exception:  # pragma: no cover - database hiccup
            return
        if not doc:
            return
        state = self.states[name]
        state.paused = bool(doc.get("paused", False))
        requested = doc.get("run_requested_at")
        if requested and (state.last_run_at is None or requested > state.last_run_at):
            state.run_requested = True
            await self.col.update_one({"name": name}, {"$unset": {"run_requested_at": ""}})

    def _snapshot(self, name: str) -> dict[str, Any]:
        state = self.states[name]
        return {
            "running": state.running,
            "paused": state.paused,
            "last_run_at": state.last_run_at,
            "last_success_at": state.last_success_at,
            "last_error": state.last_error,
            "last_error_at": state.last_error_at,
            "last_duration_ms": state.last_duration_ms,
            "next_run_at": state.next_run_at,
            "run_count": state.run_count,
            "error_count": state.error_count,
            "consecutive_errors": state.consecutive_errors,
            "last_result": state.last_result,
            "host": self.host,
        }

    async def _persist(self, name: str, fields: dict[str, Any]) -> None:
        try:
            await self.col.update_one(
                {"name": name},
                {"$set": {**fields, "updated_at": utcnow()}, "$setOnInsert": {"created_at": utcnow(), "name": name}},
                upsert=True,
            )
        except Exception as exc:  # pragma: no cover - never let bookkeeping kill a job
            logger.warning("scheduler_persist_failed", job=name, error=str(exc))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in list(value.items())[:50]}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in list(value)[:50]]
    return str(value)[:200]


__all__ = ["CONTROL_POLL_SECONDS", "JobSpec", "JobState", "Scheduler"]
