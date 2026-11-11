"""Liveness/readiness probes (unauthenticated)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Response, status
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


class DetailedHealthResponse(HealthResponse):
    system: dict[str, Any]
    collections_count: int | None = None


@router.get("/health", response_model=HealthResponse, summary="Basic health probe")
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


@router.get("/health/ready", response_model=HealthResponse, summary="Kubernetes / Docker readiness probe")
async def readiness_probe(response: Response) -> HealthResponse:
    settings = get_settings()
    ping = await mongo.ping_ms() if mongo.connected else None
    if ping is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            app=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            uptime_seconds=round(metrics.uptime_seconds, 1),
            database={"connected": False, "ping_ms": None},
        )
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(metrics.uptime_seconds, 1),
        database={"connected": True, "ping_ms": ping},
    )


@router.get("/health/live", include_in_schema=False)
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/detailed", response_model=DetailedHealthResponse, summary="Detailed system & operational health")
async def detailed_health() -> DetailedHealthResponse:
    settings = get_settings()
    ping = await mongo.ping_ms() if mongo.connected else None
    coll_count: int | None = None
    if mongo.connected and mongo.db is not None:
        try:
            colls = await mongo.db.list_collection_names()
            coll_count = len(colls)
        except Exception:
            coll_count = None

    import psutil

    process = psutil.Process(os.getpid())
    mem = psutil.virtual_memory()
    sys_info = {
        "pid": os.getpid(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_rss_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "system_memory_used_pct": mem.percent,
        "threads": process.num_threads(),
    }

    return DetailedHealthResponse(
        status="ok" if ping is not None else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(metrics.uptime_seconds, 1),
        database={"connected": ping is not None, "ping_ms": ping},
        system=sys_info,
        collections_count=coll_count,
    )
