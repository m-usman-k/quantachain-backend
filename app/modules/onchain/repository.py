"""Repositories for Module 3: on-chain intelligence, whale transfers, and wallets."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.modules.onchain.models import TransferFlow, TransferRisk, Wallet, WhaleTransfer


class WhaleTransferRepository(Repository[WhaleTransfer]):
    collection_name = Collections.WHALE_TRANSFERS
    model = WhaleTransfer
    not_found_message = "Whale transfer not found"

    async def insert_transfer(self, transfer: WhaleTransfer) -> WhaleTransfer | None:
        """Insert transfer if unique by tx_hash + log_index."""
        if await self.exists({"tx_hash": transfer.tx_hash, "log_index": transfer.log_index}):
            return None
        try:
            return await self.insert(transfer)
        except DuplicateKeyError:
            return None

    async def get_by_tx_hash(self, tx_hash: str) -> list[WhaleTransfer]:
        return await self.find({"tx_hash": tx_hash.lower()}, sort=[("log_index", ASCENDING)])

    async def find_transfers(
        self,
        *,
        symbol: str | None = None,
        min_amount_usd: float | None = None,
        flow: TransferFlow | None = None,
        risk_level: TransferRisk | None = None,
        address: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WhaleTransfer], int]:
        query: dict[str, Any] = {}
        if symbol:
            query["symbol"] = symbol.upper()
        if min_amount_usd is not None:
            query["amount_usd"] = {"$gte": min_amount_usd}
        if flow:
            query["flow"] = flow
        if risk_level:
            query["risk_level"] = risk_level
        if address:
            addr = address.lower()
            query["$or"] = [{"from_address": addr}, {"to_address": addr}]
        if since:
            query["block_time"] = {"$gte": since}

        total = await self.count(query)
        items = await self.find(query, sort=[("block_time", DESCENDING)], limit=limit, skip=offset)
        return items, total

    async def get_transfers_for_address(self, address: str, limit: int = 50) -> list[WhaleTransfer]:
        addr = address.lower()
        return await self.find(
            {"$or": [{"from_address": addr}, {"to_address": addr}]},
            sort=[("block_time", DESCENDING)],
            limit=limit,
        )

    async def netflow_summary(self, symbol: str, since: datetime) -> dict[str, float]:
        """Aggregate exchange inflows, outflows, and netflows (USD)."""
        pipeline = [
            {"$match": {"symbol": symbol.upper(), "block_time": {"$gte": since}, "amount_usd": {"$ne": None}}},
            {
                "$group": {
                    "_id": "$flow",
                    "total_usd": {"$sum": "$amount_usd"},
                    "count": {"$sum": 1},
                }
            },
        ]
        rows = await self.aggregate(pipeline)
        inflow = 0.0
        outflow = 0.0
        for r in rows:
            flow_type = r.get("_id")
            val = float(r.get("total_usd", 0.0))
            if flow_type == TransferFlow.EXCHANGE_INFLOW:
                inflow += val
            elif flow_type == TransferFlow.EXCHANGE_OUTFLOW:
                outflow += val

        return {
            "inflow_usd": round(inflow, 2),
            "outflow_usd": round(outflow, 2),
            "netflow_usd": round(inflow - outflow, 2),  # positive = net inflow to exchanges (bearish signal)
        }


class WalletRepository(Repository[Wallet]):
    collection_name = Collections.WALLETS
    model = Wallet
    not_found_message = "Wallet not found"

    async def get_by_address(self, address: str) -> Wallet | None:
        return await self.find_one({"address": address.lower()})

    async def top_smart_money(self, limit: int = 50) -> list[Wallet]:
        return await self.find(
            {"smart_money_score": {"$gte": 60}},
            sort=[("smart_money_score", DESCENDING), ("total_profit_usd", DESCENDING)],
            limit=limit,
        )

    async def upsert_wallet(self, address: str, **fields: Any) -> Wallet:
        addr = address.lower()
        fields = {k: v for k, v in fields.items() if v is not None}
        fields["updated_at"] = utcnow()
        on_insert = {"address": addr, "first_seen_at": utcnow(), "created_at": utcnow()}
        return await self.upsert({"address": addr}, fields, on_insert=on_insert)


class OnchainMetricRepository:
    """Time-series rows: ``{ts, meta: {metric, symbol, chain}, value}``."""

    def __init__(self, db: Any) -> None:
        self.col = db[Collections.ONCHAIN_METRICS]

    async def append(
        self, metric: str, symbol: str, value: float, chain: str = "eth", ts: datetime | None = None
    ) -> None:
        row = {
            "ts": ts or utcnow(),
            "meta": {"metric": metric, "symbol": symbol.upper(), "chain": chain},
            "value": float(value),
        }
        await self.col.insert_one(row)

    async def recent(self, metric: str, symbol: str, since: datetime, limit: int = 500) -> list[dict[str, Any]]:
        cursor = (
            self.col.find({"meta.metric": metric, "meta.symbol": symbol.upper(), "ts": {"$gte": since}})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        rows = [row async for row in cursor]
        rows.reverse()
        return rows


__all__ = ["OnchainMetricRepository", "WalletRepository", "WhaleTransferRepository"]
