"""Liveness/readiness probes (unauthenticated)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.metrics import metrics
from app.db.mongo import mongo

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str
    uptime_seconds: float
    database: dict[str, Any]


@router.get("/health", response_model=HealthResponse, summary="Readiness probe")
async def health_check() -> HealthResponse:
    settings = get_settings()
    ping = await mongo.ping_ms() if mongo.connected else None
    return HealthResponse(
        status="ok" if ping is not None else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(metrics.uptime_seconds, 1),
        database={"connected": ping is not None, "ping_ms": ping},
    )


@router.get("/health/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "alive"}
