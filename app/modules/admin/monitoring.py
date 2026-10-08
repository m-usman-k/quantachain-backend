"""System monitoring: resource sampling, event-loop lag and automated error alerts (Module 7).

``SystemMonitor`` runs inside whichever process hosts it (API and worker each
run one). Every minute it stores a sample in the ``system_metrics`` time-series
and, in the API process, checks the 5xx error rate and emails administrators
(with a cooldown) when it crosses the configured threshold.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import socket
import sys
import time
from typing import Any

import psutil
import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.metrics import metrics
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.integrations.email import EmailClient

logger = structlog.get_logger(__name__)

SAMPLE_INTERVAL_SECONDS = 60.0
_process = psutil.Process(os.getpid())
_process.cpu_percent(interval=None)  # prime the counter


def collect_system_metrics(*, loop_lag_ms: float | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        disk_info = {"total_bytes": disk.total, "used_bytes": disk.used, "percent": disk.percent}
    except OSError:  # pragma: no cover - exotic mounts
        disk_info = {"total_bytes": None, "used_bytes": None, "percent": None}
    load_avg: list[float] | None
    try:
        load_avg = [round(v, 2) for v in os.getloadavg()]  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        load_avg = None
    with _process.oneshot():
        rss = _process.memory_info().rss
        process_cpu = _process.cpu_percent(interval=None)
        threads = _process.num_threads()
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count() or 1,
        "load_avg": load_avg,
        "memory": {"total_bytes": memory.total, "used_bytes": memory.used, "percent": memory.percent},
        "disk": disk_info,
        "process": {
            "rss_bytes": rss,
            "cpu_percent": process_cpu,
            "threads": threads,
            "uptime_seconds": round(metrics.uptime_seconds, 1),
        },
        "event_loop_lag_ms": loop_lag_ms,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "sampled_at": utcnow(),
    }


class SystemMonitor:
    def __init__(self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, *, role: str) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.role = role  # "api" | "worker"
        self.email = EmailClient(self.settings)
        self._task: asyncio.Task[None] | None = None
        self.loop_lag_ms: float | None = None
        self.last_sample: dict[str, Any] | None = None
        self.last_alert_at: float = 0.0
        self.alerts_sent = 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"system-monitor:{self.role}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self.sample()
                if self.role == "api":
                    await self.check_error_rate()
            except Exception as exc:  # pragma: no cover - monitoring must never crash the host
                logger.warning("system_monitor_failed", error=str(exc))
            await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)

    async def measure_loop_lag(self) -> float:
        started = time.perf_counter()
        await asyncio.sleep(0)
        lag = (time.perf_counter() - started) * 1000
        self.loop_lag_ms = round(lag, 3)
        return self.loop_lag_ms

    async def sample(self) -> dict[str, Any]:
        await self.measure_loop_lag()
        sample = collect_system_metrics(loop_lag_ms=self.loop_lag_ms)
        self.last_sample = sample
        doc = {
            "ts": sample["sampled_at"],
            "meta": {"host": sample["host"], "role": self.role},
            "cpu_percent": sample["cpu_percent"],
            "memory_percent": sample["memory"]["percent"],
            "process_rss_bytes": sample["process"]["rss_bytes"],
            "event_loop_lag_ms": self.loop_lag_ms,
            "requests_total": metrics.requests_total if self.role == "api" else None,
            "error_5xx": metrics.responses_by_class.get("5xx", 0) if self.role == "api" else None,
            "latency_p95_ms": metrics.latency_percentile(95) if self.role == "api" else None,
            "websocket_connections": metrics.websocket_connections if self.role == "api" else None,
            "healthy": True,
        }
        await self.db[Collections.SYSTEM_METRICS].insert_one(doc)
        return sample

    async def check_error_rate(self) -> bool:
        """Send an admin alert when too many 5xx responses happened recently."""
        errors = metrics.errors_in_window(self.settings.error_alert_window_seconds)
        if errors < self.settings.error_alert_threshold:
            return False
        if time.monotonic() - self.last_alert_at < self.settings.error_alert_cooldown_seconds:
            return False
        self.last_alert_at = time.monotonic()
        self.alerts_sent += 1
        subject = (
            f"[Quantachain] {errors} server errors in the last {self.settings.error_alert_window_seconds // 60} min"
        )
        body = (
            f"The API returned {errors} 5xx responses within {self.settings.error_alert_window_seconds} seconds "
            f"(threshold {self.settings.error_alert_threshold}).\n\n"
            f"Environment: {self.settings.environment}\nHost: {socket.gethostname()}\n"
            f"p95 latency: {metrics.latency_percentile(95)} ms\nRequests total: {metrics.requests_total}\n"
        )
        logger.error("error_rate_alert", errors=errors, window_seconds=self.settings.error_alert_window_seconds)
        await self.notify_admins(subject, body, severity="critical")
        return True

    async def notify_admins(self, subject: str, body: str, *, severity: str = "warning") -> dict[str, Any]:
        """Email configured recipients and drop an in-app notification for every admin user."""
        emailed = await self.email.send(list(self.settings.admin_alert_emails), subject, body)
        from app.modules.notifications.models import NotificationType, Severity
        from app.modules.notifications.service import NotificationService

        service = NotificationService(self.db, self.settings)
        count = await service.notify_subscribers(
            type=NotificationType.SYSTEM,
            title=subject,
            body=body[:900],
            severity=Severity(severity),
            extra_filter={"role": "admin"},
        )
        return {"email_sent": emailed, "admins_notified": count}


__all__ = ["SAMPLE_INTERVAL_SECONDS", "SystemMonitor", "collect_system_metrics"]
