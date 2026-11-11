"""Repositories for Module 6."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.modules.fraud.models import BlacklistEntry, ContractScan, FraudAlert, ProjectScore, WatchedPair


class ContractScanRepository(Repository[ContractScan]):
    collection_name = Collections.CONTRACT_SCANS
    model = ContractScan
    not_found_message = "Scan not found"

    async def latest_for_address(self, chain: str, address: str) -> ContractScan | None:
        return await self.find_one({"chain": chain, "address": address.lower()}, sort=[("created_at", DESCENDING)])

    async def count_since(self, since: datetime) -> int:
        return await self.count({"created_at": {"$gte": since}})


class FraudAlertRepository(Repository[FraudAlert]):
    collection_name = Collections.FRAUD_ALERTS
    model = FraudAlert
    not_found_message = "Alert not found"

    async def insert_if_new(self, alert: FraudAlert) -> FraudAlert | None:
        """Insert unless an alert with the same fingerprint exists (unique index). Returns ``None`` on duplicate."""
        if await self.exists({"fingerprint": alert.fingerprint}):
            return None
        try:
            return await self.insert(alert)
        except DuplicateKeyError:
            return None

    async def latest_critical(self) -> FraudAlert | None:
        return await self.find_one({"severity": "critical"}, sort=[("created_at", DESCENDING)])

    async def counts_by_severity(self, since: datetime) -> dict[str, int]:
        rows = await self.aggregate(
            [{"$match": {"created_at": {"$gte": since}}}, {"$group": {"_id": "$severity", "n": {"$sum": 1}}}]
        )
        counts = {"info": 0, "warning": 0, "critical": 0}
        for row in rows:
            counts[str(row["_id"])] = int(row["n"])
        return counts


class WatchedPairRepository(Repository[WatchedPair]):
    collection_name = Collections.WATCHED_PAIRS
    model = WatchedPair
    not_found_message = "Watched pair not found"

    async def get_pair(self, chain: str, pair_address: str) -> WatchedPair | None:
        return await self.find_one({"chain": chain, "pair_address": pair_address.lower()})

    async def active(self, limit: int = 0) -> list[WatchedPair]:
        return await self.find({"active": True}, sort=[("last_checked_at", ASCENDING)], limit=limit)

    async def upsert_pair(self, chain: str, pair_address: str, **fields: Any) -> WatchedPair:
        fields = {k: v for k, v in fields.items() if v is not None}
        on_insert = {
            "active": True,
            "auto_added": fields.pop("auto_added", False),
            "added_by": fields.pop("added_by", None),
        }
        return await self.upsert({"chain": chain, "pair_address": pair_address.lower()}, fields, on_insert=on_insert)

    async def count_active(self) -> int:
        return await self.count({"active": True})


class LiquiditySnapshotRepository:
    """Time-series rows: ``{ts, meta: {pair_address, chain}, liquidity_usd, price_usd, volume_h1, ...}``."""

    def __init__(self, db: Any) -> None:
        self.col = db[Collections.LIQUIDITY_SNAPSHOTS]

    async def append(self, row: dict[str, Any]) -> None:
        await self.col.insert_one(row)

    async def append_many(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        result = await self.col.insert_many(rows, ordered=False)
        return len(result.inserted_ids)

    async def recent(
        self, chain: str, pair_address: str, *, since: datetime, limit: int = 2000
    ) -> list[dict[str, Any]]:
        cursor = (
            self.col.find({"meta.chain": chain, "meta.pair_address": pair_address.lower(), "ts": {"$gte": since}})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        rows = [row async for row in cursor]
        rows.reverse()
        return rows

    async def latest(self, chain: str, pair_address: str) -> dict[str, Any] | None:
        return await self.col.find_one(
            {"meta.chain": chain, "meta.pair_address": pair_address.lower()}, sort=[("ts", DESCENDING)]
        )

    async def series(
        self, chain: str, pair_address: str, *, start: datetime, bucket_seconds: int, end: datetime | None = None
    ) -> list[dict[str, Any]]:
        match: dict[str, Any] = {"meta.chain": chain, "meta.pair_address": pair_address.lower(), "ts": {"$gte": start}}
        if end is not None:
            match["ts"]["$lte"] = end
        pipeline = [
            {"$match": match},
            {"$sort": {"ts": 1}},
            {
                "$group": {
                    "_id": {"$dateTrunc": {"date": "$ts", "unit": "second", "binSize": bucket_seconds}},
                    "liquidity_usd": {"$last": "$liquidity_usd"},
                    "liquidity_high": {"$max": "$liquidity_usd"},
                    "liquidity_low": {"$min": "$liquidity_usd"},
                    "price_usd": {"$last": "$price_usd"},
                    "volume_h1": {"$last": "$volume_h1"},
                    "volume_h24": {"$last": "$volume_h24"},
                    "buys_h1": {"$last": "$buys_h1"},
                    "sells_h1": {"$last": "$sells_h1"},
                    "samples": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = await self.col.aggregate(pipeline)
        return [{"ts": row["_id"], **{k: v for k, v in row.items() if k != "_id"}} async for row in cursor]

    async def count_since(self, since: datetime) -> int:
        return await self.col.count_documents({"ts": {"$gte": since}})


class ProjectScoreRepository(Repository[ProjectScore]):
    collection_name = Collections.PROJECT_SCORES
    model = ProjectScore
    not_found_message = "Project score not found"

    async def get_for(self, chain: str, address: str) -> ProjectScore | None:
        return await self.find_one({"chain": chain, "address": address})

    async def upsert_score(self, score: ProjectScore) -> ProjectScore:
        data = score.to_mongo()
        data.pop("_id", None)
        data.pop("created_at", None)
        data.pop("updated_at", None)
        return await self.upsert({"chain": score.chain, "address": score.address}, data)

    async def average(self) -> float | None:
        rows = await self.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$score"}, "n": {"$sum": 1}}}])
        if not rows or rows[0].get("avg") is None:
            return None
        return round(float(rows[0]["avg"]), 1)


class BlacklistRepository(Repository[BlacklistEntry]):
    collection_name = Collections.WALLET_BLACKLIST
    model = BlacklistEntry
    not_found_message = "Blacklist entry not found"

    async def get_by_address(self, address: str) -> BlacklistEntry | None:
        return await self.find_one({"address": address})

    async def counts_by_status(self) -> dict[str, int]:
        rows = await self.aggregate([{"$group": {"_id": "$status", "n": {"$sum": 1}}}])
        counts = {"blocked": 0, "monitoring": 0, "cleared": 0}
        for row in rows:
            counts[str(row["_id"])] = int(row["n"])
        return counts

    async def touch(self, address: str, **fields: Any) -> BlacklistEntry | None:
        return await self.update_where({"address": address}, {**fields, "updated_at": utcnow()})


__all__ = [
    "BlacklistRepository",
    "ContractScanRepository",
    "FraudAlertRepository",
    "LiquiditySnapshotRepository",
    "ProjectScoreRepository",
    "WatchedPairRepository",
]
