"""Standalone worker process: ``python -m app.worker``.

Runs every background job (price streams, news polling, on-chain scanning,
sentiment batches, forecasts, fraud scans ...) separately from the API so heavy
work never competes with request latency.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from pathlib import Path

import structlog

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.collections import ensure_schema
from app.db.mongo import mongo
from app.modules.market.seed import ensure_default_assets
from app.workers.runtime import WorkerRuntime

logger = structlog.get_logger("worker")


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, as_json=settings.log_as_json)
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    if not settings.workers_enabled:
        logger.warning("workers_disabled_exiting")
        return

    db = await mongo.connect(settings)
    await ensure_schema(db, settings)
    await ensure_default_assets(db, settings)

    runtime = WorkerRuntime(db, settings)
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows has no loop signal handlers
            loop.add_signal_handler(sig, stop.set)

    await runtime.start()
    logger.info("worker_ready", environment=settings.environment)
    try:
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runtime.stop()
        await mongo.close()
        logger.info("worker_exited")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
