"""Repositories for Module 4: sentiment snapshots and geopolitical events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.modules.sentiment.models import GeoCategory, GeoEvent


class SentimentSnapshotRepository:
    """Time-series rows: ``{ts, meta: {scope, symbol}, score, index, label, sample_size, breakdown}``."""

    def __init__(self, db: Any) -> None:
        self.col = db[Collections.SENTIMENT_SNAPSHOTS]

    async def append(self, snapshot: dict[str, Any]) -> None:
        if "ts" not in snapshot:
            snapshot["ts"] = utcnow()
        await self.col.insert_one(snapshot)

    async def latest(self, scope: str = "global", symbol: str | None = None) -> dict[str, Any] | None:
        query: dict[str, Any] = {"meta.scope": scope}
        if symbol:
            query["meta.symbol"] = symbol.upper()
        return await self.col.find_one(query, sort=[("ts", DESCENDING)])

    async def series(
        self,
        scope: str = "global",
        symbol: str | None = None,
        *,
        since: datetime,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"meta.scope": scope, "ts": {"$gte": since}}
        if symbol:
            query["meta.symbol"] = symbol.upper()
        cursor = self.col.find(query).sort("ts", DESCENDING).limit(limit)
        rows = [r async for r in cursor]
        rows.reverse()
        return rows


class GeoEventRepository(Repository[GeoEvent]):
    collection_name = Collections.GEO_EVENTS
    model = GeoEvent
    not_found_message = "Geopolitical event not found"

    async def find_events(
        self,
        *,
        category: GeoCategory | None = None,
        sector: str | None = None,
        region: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[GeoEvent], int]:
        query: dict[str, Any] = {}
        if category:
            query["category"] = category
        if sector:
            query["impacted_sectors"] = sector
        if region:
            query["regions"] = region.upper()
        if since:
            query["published_at"] = {"$gte": since}

        total = await self.count(query)
        items = await self.find(query, sort=[("published_at", DESCENDING)], limit=limit, skip=offset)
        return items, total

    async def insert_if_unique(self, event: GeoEvent) -> GeoEvent | None:
        if await self.exists({"title": event.title, "published_at": event.published_at}):
            return None
        try:
            return await self.insert(event)
        except DuplicateKeyError:
            return None


__all__ = ["GeoEventRepository", "SentimentSnapshotRepository"]
