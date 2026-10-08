"""Module 7 background work: resource sampling inside the worker process."""

from __future__ import annotations

from app.modules.admin.controls import ModelControlService
from app.modules.admin.monitoring import SystemMonitor
from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec


def register(runtime: WorkerRuntime) -> None:
    monitor = SystemMonitor(runtime.db, runtime.settings, role="worker")
    controls = ModelControlService(runtime.db)

    async def sample() -> dict[str, object]:
        sample = await monitor.sample()
        return {
            "cpu_percent": sample["cpu_percent"],
            "memory_percent": sample["memory"]["percent"],
            "loop_lag_ms": monitor.loop_lag_ms,
        }

    runtime.add_job(
        JobSpec(
            "system_metrics_sample",
            interval_seconds=60,
            func=sample,
            module="admin",
            description="Sample worker CPU / RAM / event-loop lag into the system_metrics time-series",
            jitter_seconds=5,
        )
    )
    runtime.add_job(
        JobSpec(
            "model_controls_sync",
            interval_seconds=3600,
            func=controls.ensure_known,
            module="admin",
            description="Make sure every known AI model has a control switch",
        )
    )
