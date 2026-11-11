"""Repositories for Module 9.

Concurrency model
-----------------
The paper matcher runs twice over the same orders (price-tick stream + safety
sweep) and the API places orders at the same time, so every state change is an
atomic MongoDB operation:

* ``OrderRepository.transition`` moves an order from a set of statuses to a new
  status in one ``find_one_and_update``; a second caller gets ``None`` and
  backs off. That is what makes a fill idempotent.
* Cash and positions are updated with compare-and-set loops (filter on the value
  that was read) so two fills can never both spend the same money or coins,
  and stored values stay rounded (see ``models.py``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.models import to_object_id
from app.db.repository import Repository
from app.modules.trading.models import (
    WORKING_STATUSES,
    ExchangeCredential,
    Fill,
    Order,
    OrderStatus,
    PaperAccount,
    Position,
    TradeMode,
    round_price,
    round_qty,
    round_quote,
)

CAS_ATTEMPTS = 8


class StaleWriteError(RuntimeError):
    """A compare-and-set update kept losing against concurrent writers."""


def _statuses(values: frozenset[OrderStatus] | set[OrderStatus] | list[OrderStatus]) -> list[str]:
    return [str(status) for status in values]


class OrderRepository(Repository[Order]):
    collection_name = Collections.ORDERS
    model = Order
    not_found_message = "Order not found"

    async def get_for_user(self, user_id: str, order_id: str) -> Order | None:
        return await self.find_one({"_id": to_object_id(order_id), "user_id": user_id})

    async def working(
        self, *, mode: TradeMode, symbol: str | None = None, user_id: str | None = None, limit: int = 0
    ) -> list[Order]:
        query: dict[str, Any] = {"mode": str(mode), "status": {"$in": _statuses(WORKING_STATUSES)}}
        if symbol:
            query["symbol"] = symbol.upper()
        if user_id:
            query["user_id"] = user_id
        return await self.find(query, sort=[("created_at", ASCENDING)], limit=limit)

    async def working_symbols(self, mode: TradeMode) -> set[str]:
        rows = await self.col.distinct("symbol", {"mode": str(mode), "status": {"$in": _statuses(WORKING_STATUSES)}})
        return {str(symbol) for symbol in rows}

    async def transition(
        self,
        order_id: str,
        *,
        from_statuses: frozenset[OrderStatus] | set[OrderStatus] | list[OrderStatus],
        to_status: OrderStatus,
        fields: dict[str, Any] | None = None,
    ) -> Order | None:
        """Atomically move an order between statuses. ``None`` means another writer got there first."""
        update: dict[str, Any] = {"status": str(to_status), "updated_at": utcnow(), **(fields or {})}
        raw = await self.col.find_one_and_update(
            {"_id": to_object_id(order_id), "status": {"$in": _statuses(from_statuses)}},
            {"$set": update},
            return_document=ReturnDocument.AFTER,
        )
        return self._parse(raw) if raw else None

    async def active_user_ids(self, *, mode: TradeMode, since: datetime) -> set[str]:
        rows = await self.col.distinct("user_id", {"mode": str(mode), "created_at": {"$gte": since}})
        return {str(user_id) for user_id in rows}


class FillRepository(Repository[Fill]):
    collection_name = Collections.FILLS
    model = Fill
    not_found_message = "Fill not found"


class PositionRepository(Repository[Position]):
    collection_name = Collections.POSITIONS
    model = Position
    not_found_message = "Position not found"

    @staticmethod
    def _key(user_id: str, mode: TradeMode, symbol: str) -> dict[str, Any]:
        return {"user_id": user_id, "mode": str(mode), "symbol": symbol.upper()}

    async def get_position(self, user_id: str, mode: TradeMode, symbol: str) -> Position | None:
        return await self.find_one(self._key(user_id, mode, symbol))

    async def open_for_user(self, user_id: str, mode: TradeMode) -> list[Position]:
        return await self.find(
            {"user_id": user_id, "mode": str(mode), "quantity": {"$gt": 0}}, sort=[("symbol", ASCENDING)]
        )

    async def open_user_ids(self, mode: TradeMode) -> set[str]:
        rows = await self.col.distinct("user_id", {"mode": str(mode), "quantity": {"$gt": 0}})
        return {str(user_id) for user_id in rows}

    async def apply_buy(self, user_id: str, mode: TradeMode, symbol: str, quantity: float, price: float) -> Position:
        """Add ``quantity`` bought at ``price`` using a weighted-average entry price."""
        key = self._key(user_id, mode, symbol)
        for _ in range(CAS_ATTEMPTS):
            existing = await self.find_one(key)
            if existing is None:
                try:
                    return await self.insert(
                        Position(
                            user_id=user_id,
                            mode=mode,
                            symbol=symbol.upper(),
                            quantity=round_qty(quantity),
                            avg_entry_price=round_price(price),
                        )
                    )
                except DuplicateKeyError:
                    continue
            new_qty = round_qty(existing.quantity + quantity)
            cost = existing.quantity * existing.avg_entry_price + quantity * price
            new_avg = round_price(cost / new_qty) if new_qty > 0 else round_price(price)
            raw = await self.col.find_one_and_update(
                {**key, "quantity": existing.quantity},
                {"$set": {"quantity": new_qty, "avg_entry_price": new_avg, "updated_at": utcnow()}},
                return_document=ReturnDocument.AFTER,
            )
            if raw:
                return self._parse(raw)
        raise StaleWriteError(f"Position {symbol} for user {user_id} is being updated concurrently")

    async def apply_sell(
        self, user_id: str, mode: TradeMode, symbol: str, quantity: float, price: float, fee: float
    ) -> tuple[Position, float] | None:
        """Remove ``quantity`` sold at ``price``. Returns ``(position, realized_pnl)`` or ``None`` when
        the position is too small. A position that reaches zero is kept (quantity 0) so its lifetime
        realised PnL survives; the API hides flat positions."""
        key = self._key(user_id, mode, symbol)
        for _ in range(CAS_ATTEMPTS):
            existing = await self.find_one(key)
            if existing is None or existing.quantity + 1e-12 < quantity:
                return None
            realized = round_quote((price - existing.avg_entry_price) * quantity - fee)
            new_qty = round_qty(existing.quantity - quantity)
            flat = new_qty <= 0
            raw = await self.col.find_one_and_update(
                {**key, "quantity": existing.quantity},
                {
                    "$set": {
                        "quantity": 0.0 if flat else new_qty,
                        "avg_entry_price": 0.0 if flat else existing.avg_entry_price,
                        "realized_pnl": round_quote(existing.realized_pnl + realized),
                        "updated_at": utcnow(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if raw:
                return self._parse(raw), realized
        raise StaleWriteError(f"Position {symbol} for user {user_id} is being updated concurrently")


class PaperAccountRepository(Repository[PaperAccount]):
    collection_name = Collections.PAPER_ACCOUNTS
    model = PaperAccount
    not_found_message = "Paper account not found"

    async def get_for_user(self, user_id: str) -> PaperAccount | None:
        return await self.find_one({"user_id": user_id})

    async def get_or_create(self, user_id: str, *, quote_asset: str, starting_balance: float) -> PaperAccount:
        existing = await self.get_for_user(user_id)
        if existing is not None:
            return existing
        account = PaperAccount(
            user_id=user_id,
            quote_asset=quote_asset,
            cash_balance=round_quote(starting_balance),
            starting_balance=round_quote(starting_balance),
            equity_high_water=round_quote(starting_balance),
        )
        try:
            return await self.insert(account)
        except DuplicateKeyError:
            return await self.get_for_user(user_id)  # type: ignore[return-value]

    async def adjust(
        self,
        user_id: str,
        *,
        cash_delta: float,
        realized_delta: float = 0.0,
        fee_delta: float = 0.0,
        require_cash: bool = True,
    ) -> PaperAccount | None:
        """Move cash in or out. Returns ``None`` when ``require_cash`` and the balance would go negative."""
        for _ in range(CAS_ATTEMPTS):
            account = await self.get_for_user(user_id)
            if account is None:
                return None
            new_cash = round_quote(account.cash_balance + cash_delta)
            if require_cash and new_cash < 0:
                return None
            raw = await self.col.find_one_and_update(
                {"user_id": user_id, "cash_balance": account.cash_balance},
                {
                    "$set": {
                        "cash_balance": new_cash,
                        "realized_pnl": round_quote(account.realized_pnl + realized_delta),
                        "fees_paid": round_quote(account.fees_paid + fee_delta),
                        "updated_at": utcnow(),
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if raw:
                return self._parse(raw)
        raise StaleWriteError(f"Paper account for user {user_id} is being updated concurrently")

    async def reset(self, user_id: str, *, quote_asset: str, starting_balance: float) -> PaperAccount:
        now = utcnow()
        balance = round_quote(starting_balance)
        raw = await self.col.find_one_and_update(
            {"user_id": user_id},
            {
                "$set": {
                    "quote_asset": quote_asset,
                    "cash_balance": balance,
                    "starting_balance": balance,
                    "realized_pnl": 0.0,
                    "fees_paid": 0.0,
                    "reset_at": now,
                    "equity_high_water": balance,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return self._parse(raw)

    async def raise_high_water(self, user_id: str, equity: float) -> None:
        await self.col.update_one({"user_id": user_id}, {"$max": {"equity_high_water": round_quote(equity)}})


class ExchangeCredentialRepository(Repository[ExchangeCredential]):
    collection_name = Collections.EXCHANGE_CREDENTIALS
    model = ExchangeCredential
    not_found_message = "Exchange key not found"

    async def for_user(self, user_id: str) -> list[ExchangeCredential]:
        return await self.find({"user_id": user_id}, sort=[("exchange", ASCENDING)])

    async def get_for_user(self, user_id: str, credential_id: str) -> ExchangeCredential | None:
        return await self.find_one({"_id": to_object_id(credential_id), "user_id": user_id})

    async def get_for_exchange(self, user_id: str, exchange: str) -> ExchangeCredential | None:
        return await self.find_one({"user_id": user_id, "exchange": exchange.lower()})


class PortfolioSnapshotRepository:
    """Equity curve points in a MongoDB time-series collection (``ts`` / ``meta {user_id, mode}``)."""

    BUCKETS: dict[str, tuple[str, int]] = {
        "1m": ("minute", 1),
        "5m": ("minute", 5),
        "15m": ("minute", 15),
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day", 1),
    }

    def __init__(self, db: Any) -> None:
        self.col = db[Collections.PORTFOLIO_SNAPSHOTS]

    @staticmethod
    def _meta(user_id: str, mode: TradeMode) -> dict[str, Any]:
        return {"meta.user_id": user_id, "meta.mode": str(mode)}

    async def insert(
        self, user_id: str, mode: TradeMode, values: dict[str, float], *, at: datetime | None = None
    ) -> None:
        await self.col.insert_one({"ts": at or utcnow(), "meta": {"user_id": user_id, "mode": str(mode)}, **values})

    async def latest(self, user_id: str, mode: TradeMode) -> dict[str, Any] | None:
        return await self.col.find_one(self._meta(user_id, mode), sort=[("ts", DESCENDING)])

    async def at_or_before(self, user_id: str, mode: TradeMode, at: datetime) -> dict[str, Any] | None:
        """Snapshot closest to ``at`` from below; falls back to the oldest one after it."""
        row = await self.col.find_one({**self._meta(user_id, mode), "ts": {"$lte": at}}, sort=[("ts", DESCENDING)])
        if row is None:
            row = await self.col.find_one({**self._meta(user_id, mode), "ts": {"$gt": at}}, sort=[("ts", ASCENDING)])
        return row

    async def history(
        self, user_id: str, mode: TradeMode, *, start: datetime, interval: str, end: datetime | None = None
    ) -> list[dict[str, Any]]:
        unit, bin_size = self.BUCKETS[interval]
        time_filter: dict[str, Any] = {"$gte": start}
        if end is not None:
            time_filter["$lte"] = end
        pipeline = [
            {"$match": {**self._meta(user_id, mode), "ts": time_filter}},
            {"$sort": {"ts": 1}},
            {
                "$group": {
                    "_id": {"$dateTrunc": {"date": "$ts", "unit": unit, "binSize": bin_size}},
                    "equity": {"$last": "$equity"},
                    "cash": {"$last": "$cash"},
                    "positions_value": {"$last": "$positions_value"},
                    "unrealized_pnl": {"$last": "$unrealized_pnl"},
                    "realized_pnl": {"$last": "$realized_pnl"},
                    "high": {"$max": "$equity"},
                    "low": {"$min": "$equity"},
                    "samples": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = await self.col.aggregate(pipeline)
        return [{"ts": row["_id"], **{k: v for k, v in row.items() if k != "_id"}} async for row in cursor]

    async def delete_for_user(self, user_id: str, mode: TradeMode) -> int:
        result = await self.col.delete_many(self._meta(user_id, mode))
        return result.deleted_count


__all__ = [
    "CAS_ATTEMPTS",
    "ExchangeCredentialRepository",
    "FillRepository",
    "OrderRepository",
    "PaperAccountRepository",
    "PortfolioSnapshotRepository",
    "PositionRepository",
    "StaleWriteError",
]
