"""Quantachain API application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.db.collections import ensure_schema
from app.db.mongo import mongo
from app.modules.market.seed import ensure_default_assets

logger = structlog.get_logger(__name__)

API_DESCRIPTION = """
AI-powered crypto intelligence platform backend.

**Modules**

1. Authentication & access control - OAuth 2.0, JWT, TOTP MFA, RBAC, tamper-evident audit trail
2. Multi-source data aggregation - Binance / CoinGecko prices, NewsAPI & RSS, social signals
3. On-chain intelligence - whale tracking, wallet scoring, smart-money clusters
4. Sentiment & geopolitical analysis - GPT-4o scoring, NER, price correlation
5. AI prediction & forecasting - LSTM / statistical ensemble, volatility, trade signals
6. Risk assessment & fraud detection - contract scanning, rug-pull & pump-and-dump detection
7. Admin dashboard & monitoring - system metrics, model controls, exportable logs
8. Interactive research portal - charts, PDF/CSV reports, AI research assistant
9. Trading & order execution - paper and live (CCXT) trading with portfolio tracking
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, as_json=settings.log_as_json)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)

    db = await mongo.connect(settings)
    await ensure_schema(db, settings)
    await ensure_default_assets(db, settings)

    runtime = None
    if settings.run_workers_in_api and not settings.is_test:
        from app.workers.runtime import WorkerRuntime

        runtime = WorkerRuntime(db, settings)
        await runtime.start()
        app.state.worker_runtime = runtime

    from app.api.ws import ws_hub

    await ws_hub.start(db, settings)

    from app.modules.admin.controls import ModelControlService
    from app.modules.admin.monitoring import SystemMonitor

    await ModelControlService(db).ensure_known()
    monitor = SystemMonitor(db, settings, role="api")
    app.state.system_monitor = monitor
    if not settings.is_test:
        await monitor.start()

    logger.info("application_started", environment=settings.environment, version=settings.app_version)
    try:
        yield
    finally:
        await monitor.stop()
        await ws_hub.stop()
        if runtime is not None:
            await runtime.stop()
        await mongo.close()
        logger.info("application_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=API_DESCRIPTION,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness"},
            {"name": "auth", "description": "Module 1 - authentication, MFA, OAuth, API keys, audit"},
            {"name": "market", "description": "Module 2 - prices, candles, news, social, ingestion"},
            {"name": "onchain", "description": "Module 3 - whale transfers, wallets, clusters"},
            {"name": "sentiment", "description": "Module 4 - sentiment, NER, geopolitical impact"},
            {"name": "prediction", "description": "Module 5 - forecasts, volatility, signals, models"},
            {"name": "fraud", "description": "Module 6 - contract scans, liquidity monitoring, alerts"},
            {"name": "admin", "description": "Module 7 - system monitoring and administration"},
            {"name": "research", "description": "Module 8 - charts, reports, AI research assistant"},
            {"name": "trading", "description": "Module 9 - orders, portfolio, exchange keys"},
            {"name": "notifications", "description": "In-app notifications and push devices"},
        ],
    )
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    from app.api.ws import router as ws_router

    app.include_router(ws_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "api": settings.api_v1_prefix,
        }

    return app


app = create_app()
