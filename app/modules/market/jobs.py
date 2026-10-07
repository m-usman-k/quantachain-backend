"""Registers Module 2 background work with the worker runtime."""

from __future__ import annotations

from app.modules.market.ingestion import MarketIngestion
from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec


def register(runtime: WorkerRuntime) -> None:
    settings = runtime.settings
    ingestion = MarketIngestion(runtime.db, settings, runtime.tracker, runtime.bus)
    ingestion.register_sources()

    runtime.add_stream("binance_ws", ingestion.run_binance_stream, module="market", enabled=settings.binance_ws_enabled)
    runtime.add_job(
        JobSpec(
            "market_backfill",
            interval_seconds=6 * 3600,
            func=ingestion.backfill_candles,
            module="market",
            description="Back-fill historical candles for every tracked symbol and interval",
            run_on_start=settings.backfill_on_start,
            timeout_seconds=900,
        )
    )
    runtime.add_job(
        JobSpec(
            "market_candle_sync",
            interval_seconds=600,
            func=ingestion.sync_candles,
            module="market",
            description="Heal gaps in recent candles and rebuild open buckets",
            run_on_start=False,
            timeout_seconds=600,
        )
    )
    runtime.add_job(
        JobSpec(
            "coingecko_markets",
            interval_seconds=settings.coingecko_poll_seconds,
            func=ingestion.poll_coingecko,
            module="market",
            description="Market caps, volumes, dominance and Fear & Greed",
            jitter_seconds=5,
            timeout_seconds=120,
        )
    )
    runtime.add_job(
        JobSpec(
            "news_fetch",
            interval_seconds=settings.news_poll_seconds,
            func=ingestion.poll_news,
            module="market",
            description="Fetch and deduplicate NewsAPI + RSS articles",
            jitter_seconds=10,
            timeout_seconds=240,
        )
    )
    runtime.add_job(
        JobSpec(
            "social_fetch",
            interval_seconds=settings.social_poll_seconds,
            func=ingestion.poll_social,
            module="market",
            description="Fetch and deduplicate Reddit posts",
            jitter_seconds=10,
            enabled=settings.reddit_enabled,
            timeout_seconds=240,
        )
    )
