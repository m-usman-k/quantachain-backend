"""Module 2 ingestion workers.

* Binance WebSocket -> raw ticks (time-series) + 1m candles rolled up to 5m/15m/1h/4h/1d
* CoinGecko / alternative.me -> market overview, per-asset stats, Fear & Greed
* NewsAPI + RSS -> deduplicated, sanitised news articles awaiting sentiment analysis
* Reddit -> deduplicated social signals
* REST back-fill and periodic candle sync so charts survive disconnects
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import datetime, timedelta
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.events import EventBus, Topics, event_bus
from app.core.exceptions import ExternalServiceError
from app.core.timeutils import from_ms, utcnow
from app.integrations.binance import BinanceClient, BinanceStream, Kline, Tick
from app.integrations.coingecko import CoinGeckoClient, FearGreedClient
from app.integrations.news import (
    NewsApiClient,
    RawArticle,
    RawSocialPost,
    RedditClient,
    RssClient,
    content_hash,
    url_hash,
)
from app.modules.market.models import Candle, NewsArticle, SocialPost, SourceStatus
from app.modules.market.repository import (
    AssetRepository,
    AssetStatsRepository,
    CandleRepository,
    MarketGlobalRepository,
    NewsRepository,
    SocialRepository,
    TickRepository,
)
from app.modules.market.symbols import SymbolDetector
from app.workers.sources import SourceTracker

logger = structlog.get_logger(__name__)

BACKFILL_PLAN: dict[str, int] = {"1m": 1000, "5m": 1000, "15m": 1000, "1h": 1000, "4h": 1000, "1d": 1000}
SYNC_LIMIT = 60
TICK_FLUSH_SECONDS = 2.0
PRICE_WRITE_THROTTLE = 2.0

SOURCE_DEFS: dict[str, dict[str, str]] = {
    "binance_ws": {
        "label": "Binance WebSocket",
        "kind": "websocket",
        "engine": "Price Engine",
        "description": "Live mini-ticker and 1m klines for tracked pairs",
    },
    "coingecko": {
        "label": "CoinGecko API",
        "kind": "rest",
        "engine": "Market Overview",
        "description": "Market caps, volumes, dominance and Fear & Greed",
    },
    "newsapi": {
        "label": "NewsAPI",
        "kind": "rest",
        "engine": "Sentiment Engine",
        "description": "Global crypto & geopolitical headlines",
    },
    "rss": {
        "label": "Crypto RSS Feeds",
        "kind": "rss",
        "engine": "Sentiment Engine",
        "description": "CoinDesk, Cointelegraph, Decrypt, The Block",
    },
    "reddit": {
        "label": "Reddit Stream",
        "kind": "rest",
        "engine": "Social Sentiment",
        "description": "New posts from crypto subreddits",
    },
}


class MarketIngestion:
    def __init__(
        self,
        db: AsyncDatabase[dict[str, Any]],
        settings: Settings | None = None,
        tracker: SourceTracker | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.tracker = tracker or SourceTracker(db, self.settings)
        self.bus = bus or event_bus
        self.binance = BinanceClient(self.settings)
        self.coingecko = CoinGeckoClient(self.settings)
        self.fear_greed = FearGreedClient(self.settings)
        self.newsapi = NewsApiClient(self.settings)
        self.rss = RssClient()
        self.reddit = RedditClient()
        self.assets = AssetRepository(db)
        self.candles = CandleRepository(db)
        self.ticks = TickRepository(db)
        self.stats = AssetStatsRepository(db)
        self.global_repo = MarketGlobalRepository(db)
        self.news = NewsRepository(db)
        self.social = SocialRepository(db)
        self._tick_buffer: list[dict[str, Any]] = []
        self._last_price_write: dict[str, float] = {}
        self._detector: SymbolDetector | None = None
        self._detector_loaded_at = 0.0
        self._news_watermark: datetime | None = None

    def register_sources(self) -> None:
        for name, meta in SOURCE_DEFS.items():
            enabled = True
            if name == "binance_ws":
                enabled = self.settings.binance_ws_enabled
            elif name == "newsapi":
                enabled = self.newsapi.configured
            elif name == "reddit":
                enabled = self.settings.reddit_enabled
            self.tracker.register(name, enabled=enabled, **meta)

    # ---------------------------------------------------------------- utils
    async def tracked_symbols(self) -> list[str]:
        assets = await self.assets.tracked()
        return [a.symbol for a in assets] or list(self.settings.tracked_symbols)

    async def detector(self) -> SymbolDetector:
        if self._detector is None or time.monotonic() - self._detector_loaded_at > 600:
            self._detector = SymbolDetector(await self.assets.tracked())
            self._detector_loaded_at = time.monotonic()
        return self._detector

    # ------------------------------------------------------ Binance stream
    async def run_binance_stream(self) -> None:
        name = "binance_ws"
        while not self.tracker.enabled(name):
            await asyncio.sleep(5)
        symbols = await self.tracked_symbols()
        stream = BinanceStream(symbols, self.settings)
        flusher = asyncio.create_task(self._flush_loop(), name="tick-flusher")
        self.tracker.set_status(
            name, SourceStatus.CONNECTED, message=f"Binance WebSocket connected ({len(symbols)} pairs)"
        )
        try:
            async for item in stream:
                if not self.tracker.enabled(name):
                    stream.stop()
                    self.tracker.set_status(name, SourceStatus.DISABLED, message="Binance WebSocket disabled by admin")
                    break
                if isinstance(item, Tick):
                    await self._on_tick(item)
                elif isinstance(item, Kline):
                    await self._on_kline(item)
        finally:
            flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flusher
            await self._flush_ticks()
            if self.tracker.sources[name].status == SourceStatus.CONNECTED:
                self.tracker.set_status(name, SourceStatus.STOPPED)

    async def _on_tick(self, tick: Tick) -> None:
        now = utcnow()
        latency_ms = max(0.0, (now - tick.ts).total_seconds() * 1000)
        self.tracker.event("binance_ws", latency_ms=latency_ms)
        self._tick_buffer.append(
            {
                "ts": tick.ts,
                "meta": {"symbol": tick.symbol, "source": "binance"},
                "price": tick.price,
                "volume_24h": tick.volume_24h,
            }
        )
        self.bus.publish(
            Topics.PRICE_TICK,
            {
                "symbol": tick.symbol,
                "price": tick.price,
                "ts": tick.ts.isoformat(),
                "change_24h_pct": tick.change_24h_pct,
                "volume_24h": tick.volume_24h,
                "source": "binance",
            },
        )
        last = self._last_price_write.get(tick.symbol, 0.0)
        if time.monotonic() - last >= PRICE_WRITE_THROTTLE:
            self._last_price_write[tick.symbol] = time.monotonic()
            await self.stats.set_price(tick.symbol, tick.price, source="binance", at=tick.ts)
            if tick.change_24h_pct is not None:
                await self.stats.upsert_stats(
                    tick.symbol, change_24h_pct=round(tick.change_24h_pct, 4), volume_24h=tick.volume_24h
                )

    async def _on_kline(self, kline: Kline) -> None:
        candle = Candle(
            symbol=kline.symbol,
            interval=kline.interval,
            open_time=kline.open_time,
            close_time=kline.close_time,
            open=kline.open,
            high=kline.high,
            low=kline.low,
            close=kline.close,
            volume=kline.volume,
            quote_volume=kline.quote_volume,
            trades=kline.trades,
            source="binance",
            closed=kline.closed,
        )
        _, newly_closed = await self.candles.upsert_candle(candle)
        if newly_closed and kline.interval == "1m":
            await self.candles.rollup(candle)
            self.tracker.event(
                "binance_ws",
                count=0,
                candles_closed=self.tracker.sources["binance_ws"].extra.get("candles_closed", 0) + 1,
            )
        self.bus.publish(
            Topics.CANDLE,
            {
                "symbol": candle.symbol,
                "interval": candle.interval,
                "t": candle.open_time.isoformat(),
                "o": candle.open,
                "h": candle.high,
                "l": candle.low,
                "c": candle.close,
                "v": candle.volume,
                "closed": candle.closed,
            },
        )

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_FLUSH_SECONDS)
            await self._flush_ticks()

    async def _flush_ticks(self) -> None:
        if not self._tick_buffer:
            return
        batch, self._tick_buffer = self._tick_buffer, []
        try:
            await self.ticks.insert_many(batch)
        except Exception as exc:  # pragma: no cover - keep streaming through transient DB errors
            logger.warning("tick_flush_failed", error=str(exc), dropped=len(batch))

    # ----------------------------------------------------------- CoinGecko
    async def poll_coingecko(self) -> dict[str, Any]:
        name = "coingecko"
        if not self.tracker.enabled(name):
            return {"skipped": "disabled"}
        assets = await self.assets.tracked()
        ids = {a.coingecko_id: a.symbol for a in assets if a.coingecko_id}
        started = time.perf_counter()
        updated = 0
        try:
            rows = await self.coingecko.coins_markets(list(ids))
            for row in rows:
                symbol = ids.get(row.get("id"))
                if not symbol:
                    continue
                await self.stats.upsert_stats(
                    symbol,
                    name=row.get("name"),
                    change_1h_pct=row.get("price_change_percentage_1h_in_currency"),
                    change_24h_pct=row.get(
                        "price_change_percentage_24h_in_currency", row.get("price_change_percentage_24h")
                    ),
                    change_7d_pct=row.get("price_change_percentage_7d_in_currency"),
                    market_cap=row.get("market_cap"),
                    volume_24h=row.get("total_volume"),
                    rank=row.get("market_cap_rank"),
                    circulating_supply=row.get("circulating_supply"),
                    total_supply=row.get("total_supply"),
                    ath=row.get("ath"),
                    ath_change_pct=row.get("ath_change_percentage"),
                    high_24h=row.get("high_24h"),
                    low_24h=row.get("low_24h"),
                    image_url=row.get("image"),
                    stats_updated_at=utcnow(),
                )
                await self.assets.upsert_asset(symbol, image_url=row.get("image"), rank=row.get("market_cap_rank"))
                price = row.get("current_price")
                if price is not None and not self.settings.binance_ws_enabled:
                    await self.stats.set_price(symbol, float(price), source="coingecko")
                    self._tick_buffer.append(
                        {
                            "ts": utcnow(),
                            "meta": {"symbol": symbol, "source": "coingecko"},
                            "price": float(price),
                            "volume_24h": row.get("total_volume"),
                        }
                    )
                updated += 1
            await self._flush_ticks()

            global_data = await self.coingecko.global_data()
            dominance = global_data.get("market_cap_percentage", {})
            await self.global_repo.update_global(
                total_market_cap_usd=global_data.get("total_market_cap", {}).get("usd"),
                total_volume_24h_usd=global_data.get("total_volume", {}).get("usd"),
                btc_dominance_pct=dominance.get("btc"),
                eth_dominance_pct=dominance.get("eth"),
                market_cap_change_24h_pct=global_data.get("market_cap_change_percentage_24h_usd"),
                active_cryptocurrencies=global_data.get("active_cryptocurrencies"),
                markets=global_data.get("markets"),
            )
        except ExternalServiceError as exc:
            self.tracker.error(name, str(exc))
            raise
        try:
            fng = await self.fear_greed.latest()
            if fng:
                await self.global_repo.update_global(
                    fear_greed_value=fng[0]["value"],
                    fear_greed_label=fng[0]["label"],
                    fear_greed_updated_at=datetime.fromtimestamp(fng[0]["timestamp"], tz=utcnow().tzinfo),
                )
        except ExternalServiceError as exc:
            logger.warning("fear_greed_unavailable", error=str(exc))
        latency = (time.perf_counter() - started) * 1000
        self.tracker.event(name, count=updated + 1, latency_ms=latency)
        self.tracker.set_status(name, SourceStatus.ACTIVE)
        self.bus.publish(Topics.MARKET_OVERVIEW, {"updated_assets": updated, "at": utcnow().isoformat()})
        await self.tracker.log(name, f"CoinGecko refreshed {updated} assets + global metrics in {latency:.0f}ms")
        return {"updated_assets": updated, "latency_ms": round(latency, 1)}

    # ---------------------------------------------------------------- news
    async def poll_news(self) -> dict[str, Any]:
        detector = await self.detector()
        since = self._news_watermark or (utcnow() - timedelta(days=2))
        summary: dict[str, Any] = {"new": 0, "duplicates": 0, "sources": {}}
        raw: list[tuple[str, RawArticle]] = []

        if self.tracker.enabled("newsapi") and self.newsapi.configured:
            started = time.perf_counter()
            try:
                items = await self.newsapi.everything(self.settings.news_query, since=since)
                raw.extend(("newsapi", a) for a in items)
                self.tracker.event(
                    "newsapi", count=max(len(items), 1), latency_ms=(time.perf_counter() - started) * 1000
                )
                self.tracker.set_status("newsapi", SourceStatus.ACTIVE)
                summary["sources"]["newsapi"] = len(items)
            except ExternalServiceError as exc:
                self.tracker.error("newsapi", str(exc))

        if self.tracker.enabled("rss"):
            fetched = 0
            failures = 0
            for feed_url in self.settings.rss_feeds:
                started = time.perf_counter()
                try:
                    items = await self.rss.fetch(feed_url)
                    raw.extend(("rss", a) for a in items)
                    fetched += len(items)
                    self.tracker.event(
                        "rss", count=max(len(items), 1), latency_ms=(time.perf_counter() - started) * 1000
                    )
                except ExternalServiceError as exc:
                    failures += 1
                    logger.warning("rss_feed_failed", feed=feed_url, error=str(exc))
            summary["sources"]["rss"] = fetched
            if failures and failures == len(self.settings.rss_feeds):
                self.tracker.error("rss", "All RSS feeds failed")
            else:
                self.tracker.set_status("rss", SourceStatus.DEGRADED if failures else SourceStatus.ACTIVE)

        newest = since
        for origin, article in raw:
            doc = NewsArticle(
                url=article.url,
                url_hash=url_hash(article.url),
                title=article.title,
                summary=article.summary,
                content=article.content,
                source_name=article.source_name,
                author=article.author,
                image_url=article.image_url,
                published_at=article.published_at,
                fetched_at=utcnow(),
                language=article.language,
                origin=origin,
                symbols=detector.detect(article.title, article.summary),
            )
            if await self.news.insert_if_new(doc):
                summary["new"] += 1
                newest = max(newest, article.published_at)
                self.bus.publish(
                    Topics.NEWS,
                    {
                        "id": doc.id,
                        "title": doc.title,
                        "source": doc.source_name,
                        "symbols": doc.symbols,
                        "published_at": doc.published_at.isoformat(),
                    },
                )
            else:
                summary["duplicates"] += 1
        self._news_watermark = min(newest, utcnow()) - timedelta(minutes=30)
        if summary["new"]:
            await self.tracker.log(
                "rss" if "rss" in summary["sources"] else "newsapi",
                f"Ingested {summary['new']} new articles ({summary['duplicates']} duplicates skipped)",
            )
        return summary

    # -------------------------------------------------------------- social
    async def poll_social(self) -> dict[str, Any]:
        name = "reddit"
        if not self.tracker.enabled(name):
            return {"skipped": "disabled"}
        detector = await self.detector()
        summary: dict[str, Any] = {"new": 0, "duplicates": 0, "fetched": 0}
        failures = 0
        for subreddit in self.settings.reddit_subreddits:
            started = time.perf_counter()
            try:
                posts = await self.reddit.new_posts(subreddit, limit=50)
            except ExternalServiceError as exc:
                failures += 1
                logger.warning("reddit_fetch_failed", subreddit=subreddit, error=str(exc))
                continue
            self.tracker.event(name, count=max(len(posts), 1), latency_ms=(time.perf_counter() - started) * 1000)
            summary["fetched"] += len(posts)
            for post in posts:
                if await self._store_social(post, detector):
                    summary["new"] += 1
                else:
                    summary["duplicates"] += 1
        if failures == len(self.settings.reddit_subreddits):
            self.tracker.error(name, "All subreddit fetches failed")
        else:
            self.tracker.set_status(name, SourceStatus.DEGRADED if failures else SourceStatus.ACTIVE)
        if summary["new"]:
            await self.tracker.log(
                name, f"Ingested {summary['new']} social posts ({summary['duplicates']} duplicates skipped)"
            )
        return summary

    async def _store_social(self, post: RawSocialPost, detector: SymbolDetector) -> bool:
        text_for_hash = f"{post.title} {post.text}".strip()
        if len(text_for_hash) < 12:
            return False
        doc = SocialPost(
            platform=post.platform,
            external_id=post.external_id,
            content_hash=content_hash(text_for_hash),
            title=post.title,
            text=post.text,
            url=post.url,
            author=post.author,
            posted_at=post.posted_at,
            score=post.score,
            comments=post.comments,
            community=post.community,
            symbols=detector.detect(post.title, post.text),
            extra=post.extra,
        )
        return await self.social.insert_if_new(doc)

    # ------------------------------------------------------------ candles
    async def backfill_candles(self) -> dict[str, Any]:
        return await self._sync_candles(BACKFILL_PLAN, label="backfill")

    async def sync_candles(self) -> dict[str, Any]:
        return await self._sync_candles({interval: SYNC_LIMIT for interval in BACKFILL_PLAN}, label="sync")

    async def _sync_candles(self, plan: dict[str, int], *, label: str) -> dict[str, Any]:
        symbols = await self.tracked_symbols()
        written = 0
        failures: list[str] = []
        for symbol in symbols:
            try:
                for interval, limit in plan.items():
                    klines = await self.binance.fetch_klines(symbol, interval, limit=limit)
                    candles = [
                        Candle(
                            symbol=k.symbol,
                            interval=k.interval,
                            open_time=k.open_time,
                            close_time=k.close_time,
                            open=k.open,
                            high=k.high,
                            low=k.low,
                            close=k.close,
                            volume=k.volume,
                            quote_volume=k.quote_volume,
                            trades=k.trades,
                            source="binance",
                            closed=k.closed,
                        )
                        for k in klines
                    ]
                    written += await self.candles.bulk_upsert(candles)
                    if klines and interval == "1m":
                        last = klines[-1]
                        await self.stats.set_price(
                            symbol, last.close, source="binance", at=last.close_time if last.closed else utcnow()
                        )
            except ExternalServiceError as exc:
                failures.append(symbol)
                logger.warning("binance_klines_failed", symbol=symbol, error=str(exc))
                written += await self._coingecko_fallback(symbol)
            await self.candles.rebuild_open_buckets(symbol)
        await self.tracker.log(
            "binance_ws",
            f"Candle {label}: {written} candles for {len(symbols)} symbols"
            + (f", fallback used for {failures}" if failures else ""),
        )
        return {"candles": written, "symbols": len(symbols), "fallback_symbols": failures}

    async def _coingecko_fallback(self, symbol: str) -> int:
        """Build hourly and daily candles from CoinGecko prices when Binance is unreachable."""
        asset = await self.assets.get_by_symbol(symbol)
        if asset is None or not asset.coingecko_id:
            return 0
        written = 0
        try:
            for days, interval in ((90, "1h"), (365, "1d")):
                chart = await self.coingecko.market_chart(asset.coingecko_id, days=days)
                prices = chart.get("prices", [])
                volumes = {int(v[0]): float(v[1]) for v in chart.get("total_volumes", [])}
                candles = []
                for ts_ms, price in prices:
                    open_time = from_ms(int(ts_ms))
                    candles.append(
                        Candle(
                            symbol=symbol,
                            interval=interval,
                            open_time=open_time,
                            close_time=open_time,
                            open=float(price),
                            high=float(price),
                            low=float(price),
                            close=float(price),
                            volume=0.0,
                            quote_volume=volumes.get(int(ts_ms), 0.0),
                            trades=0,
                            source="coingecko",
                            closed=True,
                        )
                    )
                written += await self.candles.bulk_upsert(candles)
        except ExternalServiceError as exc:
            logger.warning("coingecko_fallback_failed", symbol=symbol, error=str(exc))
        return written


__all__ = ["BACKFILL_PLAN", "SOURCE_DEFS", "MarketIngestion"]
