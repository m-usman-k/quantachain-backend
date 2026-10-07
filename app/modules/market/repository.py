"""Repositories for market data."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument, UpdateOne
from pymongo.errors import BulkWriteError, DuplicateKeyError

from app.core.timeutils import floor_time, interval_delta, utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.modules.market.models import (
    AnalysisStatus,
    Asset,
    AssetStats,
    Candle,
    IngestionSource,
    MarketGlobal,
    NewsArticle,
    SocialPost,
    TextAnalysis,
)

ROLLUP_INTERVALS: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d")


class AssetRepository(Repository[Asset]):
    collection_name = Collections.ASSETS
    model = Asset
    not_found_message = "Asset not found"

    async def get_by_symbol(self, symbol: str) -> Asset | None:
        return await self.find_one({"symbol": symbol.upper()})

    async def tracked(self) -> list[Asset]:
        return await self.find({"is_tracked": True}, sort=[("rank", ASCENDING), ("symbol", ASCENDING)])

    async def upsert_asset(self, symbol: str, **fields: Any) -> Asset:
        fields = {k: v for k, v in fields.items() if v is not None}
        return await self.upsert({"symbol": symbol.upper()}, fields, on_insert={"is_tracked": True})


class CandleRepository(Repository[Candle]):
    collection_name = Collections.CANDLES
    model = Candle

    @staticmethod
    def _key(candle: Candle) -> dict[str, Any]:
        return {"symbol": candle.symbol, "interval": candle.interval, "open_time": candle.open_time}

    async def upsert_candle(self, candle: Candle) -> tuple[Candle, bool]:
        """Insert or replace one candle. Returns ``(candle, newly_closed)``.

        ``newly_closed`` is True the first time a closed candle is stored, which is
        the only moment it should be rolled up into larger intervals.
        """
        data = candle.to_mongo()
        data.pop("_id", None)
        created_at = data.pop("created_at")
        data["updated_at"] = utcnow()
        previous = await self.col.find_one_and_update(
            self._key(candle),
            {"$set": data, "$setOnInsert": {"created_at": created_at}},
            upsert=True,
            return_document=ReturnDocument.BEFORE,
        )
        was_closed = bool(previous and previous.get("closed"))
        return candle, candle.closed and not was_closed

    async def bulk_upsert(self, candles: list[Candle]) -> int:
        if not candles:
            return 0
        ops = []
        for candle in candles:
            data = candle.to_mongo()
            data.pop("_id", None)
            created_at = data.pop("created_at")
            data["updated_at"] = utcnow()
            ops.append(
                UpdateOne(self._key(candle), {"$set": data, "$setOnInsert": {"created_at": created_at}}, upsert=True)
            )
        try:
            result = await self.col.bulk_write(ops, ordered=False)
        except BulkWriteError as exc:  # pragma: no cover - concurrent writers
            return int(exc.details.get("nUpserted", 0) + exc.details.get("nModified", 0))
        return result.upserted_count + result.modified_count

    async def rollup(self, minute: Candle, intervals: tuple[str, ...] = ROLLUP_INTERVALS) -> None:
        """Fold a *closed* 1m candle into every larger interval bucket."""
        now = utcnow()
        ops = []
        for interval in intervals:
            bucket_open = floor_time(minute.open_time, interval)
            bucket_close = bucket_open + interval_delta(interval) - timedelta(milliseconds=1)
            is_last_minute = minute.close_time >= bucket_close - timedelta(seconds=1)
            ops.append(
                UpdateOne(
                    {"symbol": minute.symbol, "interval": interval, "open_time": bucket_open},
                    {
                        "$setOnInsert": {
                            "open": minute.open,
                            "close_time": bucket_close,
                            "source": minute.source,
                            "created_at": now,
                        },
                        "$max": {"high": minute.high},
                        "$min": {"low": minute.low},
                        "$set": {"close": minute.close, "closed": is_last_minute, "updated_at": now},
                        "$inc": {"volume": minute.volume, "quote_volume": minute.quote_volume, "trades": minute.trades},
                    },
                    upsert=True,
                )
            )
        if ops:
            await self.col.bulk_write(ops, ordered=False)

    async def rebuild_open_buckets(self, symbol: str, intervals: tuple[str, ...] = ROLLUP_INTERVALS) -> int:
        """Recompute the currently open bucket of each interval from 1m candles (self-healing)."""
        now = utcnow()
        rebuilt = 0
        for interval in intervals:
            bucket_open = floor_time(now, interval)
            bucket_close = bucket_open + interval_delta(interval) - timedelta(milliseconds=1)
            pipeline = [
                {
                    "$match": {
                        "symbol": symbol,
                        "interval": "1m",
                        "open_time": {"$gte": bucket_open, "$lte": bucket_close},
                    }
                },
                {"$sort": {"open_time": 1}},
                {
                    "$group": {
                        "_id": None,
                        "open": {"$first": "$open"},
                        "high": {"$max": "$high"},
                        "low": {"$min": "$low"},
                        "close": {"$last": "$close"},
                        "volume": {"$sum": "$volume"},
                        "quote_volume": {"$sum": "$quote_volume"},
                        "trades": {"$sum": "$trades"},
                        "source": {"$first": "$source"},
                    }
                },
            ]
            rows = await self.aggregate(pipeline)
            if not rows:
                continue
            row = rows[0]
            row.pop("_id", None)
            await self.col.update_one(
                {"symbol": symbol, "interval": interval, "open_time": bucket_open},
                {
                    "$set": {**row, "close_time": bucket_close, "closed": False, "updated_at": now},
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            rebuilt += 1
        return rebuilt

    async def range(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Candle]:
        query: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval}
        time_filter: dict[str, Any] = {}
        if start is not None:
            time_filter["$gte"] = start
        if end is not None:
            time_filter["$lte"] = end
        if time_filter:
            query["open_time"] = time_filter
        rows = await self.find(query, sort=[("open_time", DESCENDING)], limit=limit)
        rows.reverse()
        return rows

    async def latest(self, symbol: str, interval: str) -> Candle | None:
        return await self.find_one({"symbol": symbol.upper(), "interval": interval}, sort=[("open_time", DESCENDING)])

    async def closes(self, symbol: str, interval: str, limit: int) -> list[tuple[datetime, float]]:
        cursor = (
            self.col.find({"symbol": symbol.upper(), "interval": interval}, {"open_time": 1, "close": 1})
            .sort("open_time", DESCENDING)
            .limit(limit)
        )
        rows = [(row["open_time"], float(row["close"])) async for row in cursor]
        rows.reverse()
        return rows

    async def count_for(self, symbol: str, interval: str) -> int:
        return await self.count({"symbol": symbol.upper(), "interval": interval})


class TickRepository:
    """Raw price ticks live in a MongoDB time-series collection."""

    def __init__(self, db: Any) -> None:
        self.col = db[Collections.PRICE_TICKS]

    async def insert_many(self, ticks: list[dict[str, Any]]) -> int:
        if not ticks:
            return 0
        result = await self.col.insert_many(ticks, ordered=False)
        return len(result.inserted_ids)

    async def latest(self, symbol: str) -> dict[str, Any] | None:
        return await self.col.find_one({"meta.symbol": symbol.upper()}, sort=[("ts", DESCENDING)])

    async def series(
        self, symbol: str, *, start: datetime, end: datetime | None = None, bucket_seconds: int = 60
    ) -> list[dict[str, Any]]:
        match: dict[str, Any] = {"meta.symbol": symbol.upper(), "ts": {"$gte": start}}
        if end is not None:
            match["ts"]["$lte"] = end
        pipeline = [
            {"$match": match},
            {"$sort": {"ts": 1}},
            {
                "$group": {
                    "_id": {"$dateTrunc": {"date": "$ts", "unit": "second", "binSize": bucket_seconds}},
                    "price": {"$last": "$price"},
                    "high": {"$max": "$price"},
                    "low": {"$min": "$price"},
                    "ticks": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = await self.col.aggregate(pipeline)
        return [{"ts": row["_id"], **{k: v for k, v in row.items() if k != "_id"}} async for row in cursor]

    async def count_since(self, since: datetime) -> int:
        return await self.col.count_documents({"ts": {"$gte": since}})


class AssetStatsRepository(Repository[AssetStats]):
    collection_name = Collections.ASSET_STATS
    model = AssetStats

    async def get_by_symbol(self, symbol: str) -> AssetStats | None:
        return await self.find_one({"symbol": symbol.upper()})

    async def upsert_stats(self, symbol: str, **fields: Any) -> AssetStats:
        fields = {k: v for k, v in fields.items() if v is not None}
        return await self.upsert({"symbol": symbol.upper()}, fields)

    async def set_price(self, symbol: str, price: float, *, source: str, at: datetime | None = None) -> None:
        await self.col.update_one(
            {"symbol": symbol.upper()},
            {
                "$set": {
                    "price": price,
                    "price_source": source,
                    "price_updated_at": at or utcnow(),
                    "updated_at": utcnow(),
                },
                "$setOnInsert": {"created_at": utcnow()},
            },
            upsert=True,
        )

    async def ranked(
        self, *, sort_field: str, direction: int, limit: int, symbols: list[str] | None = None
    ) -> list[AssetStats]:
        query: dict[str, Any] = {sort_field: {"$ne": None}}
        if symbols is not None:
            query["symbol"] = {"$in": [s.upper() for s in symbols]}
        return await self.find(query, sort=[(sort_field, direction)], limit=limit)


class MarketGlobalRepository(Repository[MarketGlobal]):
    collection_name = Collections.MARKET_GLOBAL
    model = MarketGlobal
    GLOBAL_ID = "global"

    async def get_global(self) -> MarketGlobal | None:
        return await self.get(self.GLOBAL_ID)

    async def update_global(self, **fields: Any) -> MarketGlobal:
        fields = {k: v for k, v in fields.items() if v is not None}
        return await self.upsert({"_id": self.GLOBAL_ID}, fields)


class NewsRepository(Repository[NewsArticle]):
    collection_name = Collections.NEWS_ARTICLES
    model = NewsArticle
    not_found_message = "Article not found"

    async def insert_if_new(self, article: NewsArticle) -> bool:
        """Insert unless an article with the same canonical URL exists. Never overwrites analysis."""
        try:
            await self.insert(article)
            return True
        except DuplicateKeyError:
            return False

    async def pending(self, limit: int) -> list[NewsArticle]:
        return await self.find(
            {"analysis.status": AnalysisStatus.PENDING}, sort=[("published_at", DESCENDING)], limit=limit
        )

    async def set_analysis(self, article_id: str, analysis: TextAnalysis) -> None:
        await self.update(article_id, {"analysis": analysis.model_dump()})

    async def recent(
        self, *, since: datetime, symbols: list[str] | None = None, analyzed_only: bool = True
    ) -> list[NewsArticle]:
        query: dict[str, Any] = {"published_at": {"$gte": since}}
        if symbols:
            query["symbols"] = {"$in": [s.upper() for s in symbols]}
        if analyzed_only:
            query["analysis.status"] = AnalysisStatus.DONE
        return await self.find(query, sort=[("published_at", DESCENDING)], limit=2000)


class SocialRepository(Repository[SocialPost]):
    collection_name = Collections.SOCIAL_POSTS
    model = SocialPost
    not_found_message = "Post not found"

    async def insert_if_new(self, post: SocialPost) -> bool:
        if await self.exists({"content_hash": post.content_hash}):
            return False
        try:
            await self.insert(post)
            return True
        except DuplicateKeyError:
            return False

    async def pending(self, limit: int) -> list[SocialPost]:
        return await self.find(
            {"analysis.status": AnalysisStatus.PENDING}, sort=[("posted_at", DESCENDING)], limit=limit
        )

    async def set_analysis(self, post_id: str, analysis: TextAnalysis) -> None:
        await self.update(post_id, {"analysis": analysis.model_dump()})

    async def recent(
        self, *, since: datetime, symbols: list[str] | None = None, analyzed_only: bool = True
    ) -> list[SocialPost]:
        query: dict[str, Any] = {"posted_at": {"$gte": since}}
        if symbols:
            query["symbols"] = {"$in": [s.upper() for s in symbols]}
        if analyzed_only:
            query["analysis.status"] = AnalysisStatus.DONE
        return await self.find(query, sort=[("posted_at", DESCENDING)], limit=5000)


class IngestionSourceRepository(Repository[IngestionSource]):
    collection_name = Collections.INGESTION_SOURCES
    model = IngestionSource
    not_found_message = "Ingestion source not found"

    async def get_by_name(self, name: str) -> IngestionSource | None:
        return await self.find_one({"name": name})

    async def upsert_source(self, name: str, **fields: Any) -> IngestionSource:
        return await self.upsert({"name": name}, fields)

    async def set_enabled(self, name: str, enabled: bool) -> IngestionSource | None:
        return await self.update_where({"name": name}, {"enabled": enabled})


__all__ = [
    "ROLLUP_INTERVALS",
    "AssetRepository",
    "AssetStatsRepository",
    "CandleRepository",
    "IngestionSourceRepository",
    "MarketGlobalRepository",
    "NewsRepository",
    "SocialRepository",
    "TickRepository",
]
