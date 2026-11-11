"""Background jobs for Module 4: pending news sentiment processing & snapshot generation."""

from __future__ import annotations

import structlog

from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec

logger = structlog.get_logger(__name__)


def register(runtime: WorkerRuntime) -> None:
    async def process_sentiment_batch() -> None:
        from app.modules.sentiment.service import SentimentService

        service = SentimentService(runtime.db, runtime.settings)
        processed = await service.process_pending_articles(limit=runtime.settings.sentiment_batch_size)
        if processed > 0:
            logger.info("sentiment_batch_processed", count=processed)

    async def generate_hourly_snapshots() -> None:
        from app.modules.sentiment.service import SentimentService

        service = SentimentService(runtime.db, runtime.settings)
        await service.generate_snapshot(scope="global")
        for sym in runtime.settings.tracked_symbols:
            try:
                await service.generate_snapshot(scope="asset", symbol=sym)
            except Exception as exc:
                logger.warning("asset_snapshot_failed", symbol=sym, error=str(exc))

    runtime.add_job(
        JobSpec(
            name="sentiment.process_batch",
            interval_seconds=runtime.settings.sentiment_poll_seconds,
            func=process_sentiment_batch,
            module="sentiment",
            description="Process unanalyzed news articles and social posts with NLP",
            run_on_start=False,
            timeout_seconds=60.0,
        )
    )

    runtime.add_job(
        JobSpec(
            name="sentiment.generate_snapshots",
            interval_seconds=runtime.settings.sentiment_snapshot_seconds,
            func=generate_hourly_snapshots,
            module="sentiment",
            description="Generate Vibe Index sentiment snapshots for global and tracked assets",
            run_on_start=False,
            timeout_seconds=60.0,
        )
    )
