"""Module 8 background work: periodic macro insights."""

from __future__ import annotations

from app.modules.admin.controls import ModelControlService
from app.modules.research.service import ResearchService
from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec


def register(runtime: WorkerRuntime) -> None:
    service = ResearchService(runtime.db, runtime.settings)
    controls = ModelControlService(runtime.db)

    async def refresh() -> dict[str, object]:
        if await controls.is_paused("insights"):
            return {"skipped": "paused"}
        results = await service.refresh_insights()
        await controls.record_run("insights", computed=len(results))
        return {"computed": len(results), "titles": [r.title for r in results]}

    runtime.add_job(
        JobSpec(
            "insights_refresh",
            interval_seconds=6 * 3600,
            func=refresh,
            module="research",
            description="Aggregate monthly and quarterly macro insights across all modules",
            jitter_seconds=60,
            timeout_seconds=600,
        )
    )
