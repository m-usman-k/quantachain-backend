"""Repositories for Module 5 (and the shared ``ReportRepository`` used by Module 8)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.pagination import PageParams
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.models import to_object_id
from app.db.repository import Repository
from app.modules.prediction.models import MLModel, Prediction, Report, ReportStatus, TradeSignal, VolatilityForecast

WEIGHTS_STATE_KEY = "prediction:weights"
ACCURACY_STATE_KEY = "prediction:accuracy"


class PredictionRepository(Repository[Prediction]):
    collection_name = Collections.PREDICTIONS
    model = Prediction
    not_found_message = "Prediction not found"

    async def latest_for(self, symbol: str, horizon: str | None = None) -> list[Prediction]:
        query: dict[str, Any] = {"symbol": symbol.upper(), "latest": True}
        if horizon:
            query["horizon"] = horizon
        return await self.find(query, sort=[("steps", 1)])

    async def latest_matrix(self, symbols: list[str]) -> dict[str, dict[str, Prediction]]:
        rows = await self.find({"symbol": {"$in": [s.upper() for s in symbols]}, "latest": True})
        matrix: dict[str, dict[str, Prediction]] = {}
        for row in rows:
            matrix.setdefault(row.symbol, {})[row.horizon] = row
        return matrix

    async def store_latest(self, prediction: Prediction) -> Prediction:
        """Insert a prediction and demote the previous ``latest`` document of the same symbol/horizon."""
        inserted = await self.insert(prediction)
        await self.col.update_many(
            {
                "symbol": prediction.symbol,
                "horizon": prediction.horizon,
                "latest": True,
                "_id": {"$ne": to_object_id(inserted.id or "")},
            },
            {"$set": {"latest": False, "updated_at": utcnow()}},
        )
        return inserted

    async def pending_evaluation(self, now: datetime, *, limit: int = 2000) -> list[Prediction]:
        return await self.find(
            {"target_time": {"$lte": now}, "evaluated": False}, sort=[("target_time", 1)], limit=limit
        )

    async def history(
        self, symbol: str, horizon: str, *, since: datetime, limit: int = 500, evaluated_only: bool = False
    ) -> list[Prediction]:
        query: dict[str, Any] = {"symbol": symbol.upper(), "horizon": horizon, "created_at": {"$gte": since}}
        if evaluated_only:
            query["evaluated"] = True
        return await self.find(query, sort=[("created_at", DESCENDING)], limit=limit)

    async def recent_hits(self, symbol: str, horizon: str, *, limit: int = 50) -> tuple[int, int]:
        """``(hits, evaluated)`` over the most recent evaluated predictions - the rolling accuracy."""
        cursor = (
            self.col.find(
                {"symbol": symbol.upper(), "horizon": horizon, "evaluated": True, "direction_hit": {"$ne": None}},
                {"direction_hit": 1},
            )
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        hits = total = 0
        async for row in cursor:
            total += 1
            hits += 1 if row.get("direction_hit") else 0
        return hits, total

    async def accuracy(
        self, *, symbol: str | None = None, since: datetime | None = None
    ) -> dict[str, dict[str, float | int]]:
        """Directional accuracy and MAPE of evaluated predictions grouped by horizon."""
        match: dict[str, Any] = {"evaluated": True, "realized_price": {"$ne": None}}
        if symbol:
            match["symbol"] = symbol.upper()
        if since is not None:
            match["created_at"] = {"$gte": since}
        pipeline = [
            {"$match": match},
            {
                "$group": {
                    "_id": "$horizon",
                    "count": {"$sum": 1},
                    "hits": {"$sum": {"$cond": [{"$eq": ["$direction_hit", True]}, 1, 0]}},
                    "mape": {"$avg": "$abs_error_pct"},
                    "avg_confidence": {"$avg": "$confidence"},
                }
            },
        ]
        out: dict[str, dict[str, float | int]] = {}
        for row in await self.aggregate(pipeline):
            out[str(row["_id"])] = {
                "count": int(row["count"]),
                "hits": int(row["hits"]),
                "mape": float(row["mape"]) if row.get("mape") is not None else 0.0,
                "avg_confidence": float(row["avg_confidence"]) if row.get("avg_confidence") is not None else 0.0,
            }
        return out


class VolatilityRepository(Repository[VolatilityForecast]):
    collection_name = Collections.VOLATILITY_FORECASTS
    model = VolatilityForecast
    not_found_message = "Volatility forecast not found"

    async def latest(self, symbol: str) -> VolatilityForecast | None:
        return await self.find_one({"symbol": symbol.upper()}, sort=[("created_at", DESCENDING)])

    async def store_latest(self, forecast: VolatilityForecast) -> VolatilityForecast:
        inserted = await self.insert(forecast)
        await self.col.update_many(
            {"symbol": forecast.symbol, "latest": True, "_id": {"$ne": to_object_id(inserted.id or "")}},
            {"$set": {"latest": False, "updated_at": utcnow()}},
        )
        return inserted

    async def series(self, symbol: str, *, since: datetime, limit: int = 500) -> list[VolatilityForecast]:
        rows = await self.find(
            {"symbol": symbol.upper(), "created_at": {"$gte": since}}, sort=[("created_at", DESCENDING)], limit=limit
        )
        rows.reverse()
        return rows


class TradeSignalRepository(Repository[TradeSignal]):
    collection_name = Collections.TRADE_SIGNALS
    model = TradeSignal
    not_found_message = "Signal not found"

    async def has_active(self, symbol: str, horizon: str, direction: str, now: datetime) -> bool:
        return await self.exists(
            {"symbol": symbol.upper(), "horizon": horizon, "direction": direction, "expires_at": {"$gt": now}}
        )

    async def active(self, symbol: str | None, now: datetime, *, limit: int = 100) -> list[TradeSignal]:
        query: dict[str, Any] = {"expires_at": {"$gt": now}}
        if symbol:
            query["symbol"] = symbol.upper()
        return await self.find(query, sort=[("created_at", DESCENDING)], limit=limit)

    async def recent(self, symbol: str | None, *, limit: int = 50) -> list[TradeSignal]:
        query: dict[str, Any] = {}
        if symbol:
            query["symbol"] = symbol.upper()
        return await self.find(query, sort=[("created_at", DESCENDING)], limit=limit)


class MLModelRepository(Repository[MLModel]):
    collection_name = Collections.ML_MODELS
    model = MLModel
    not_found_message = "Model not found"

    async def active_model(self, name: str, symbol: str, interval: str) -> MLModel | None:
        return await self.find_one(
            {"name": name, "symbol": symbol.upper(), "interval": interval, "active": True},
            sort=[("version", DESCENDING)],
        )

    async def next_version(self, name: str, symbol: str, interval: str) -> int:
        latest = await self.find_one(
            {"name": name, "symbol": symbol.upper(), "interval": interval}, sort=[("version", DESCENDING)]
        )
        return (latest.version + 1) if latest else 1

    async def register(self, model: MLModel) -> MLModel:
        """Insert a trained model and retire previous active versions for the same symbol/interval."""
        inserted = await self.insert(model)
        await self.col.update_many(
            {
                "name": model.name,
                "symbol": model.symbol,
                "interval": model.interval,
                "active": True,
                "_id": {"$ne": to_object_id(inserted.id or "")},
            },
            {"$set": {"active": False, "updated_at": utcnow()}},
        )
        return inserted

    async def list_models(self, *, active_only: bool = False) -> list[MLModel]:
        query: dict[str, Any] = {"active": True} if active_only else {}
        return await self.find(query, sort=[("symbol", 1), ("version", DESCENDING)])


class ReportRepository(Repository[Report]):
    """Shared by Modules 5 and 8: every generated report lives in ``reports``."""

    collection_name = Collections.REPORTS
    model = Report
    not_found_message = "Report not found"

    async def for_user(
        self, user_id: str, params: PageParams, *, type: str | None = None, symbol: str | None = None
    ) -> tuple[list[Report], int]:
        query: dict[str, Any] = {"user_id": user_id}
        if type:
            query["type"] = type
        if symbol:
            query["symbol"] = symbol.upper()
        return await self.paginate(query, params, sort=[("created_at", DESCENDING)])

    async def mark_ready(self, report_id: str, *, file_path: str, size_bytes: int) -> Report | None:
        return await self.update(
            report_id,
            {
                "status": ReportStatus.READY,
                "file_path": file_path,
                "size_bytes": size_bytes,
                "error": None,
                "completed_at": utcnow(),
            },
        )

    async def mark_failed(self, report_id: str, error: str) -> Report | None:
        return await self.update(
            report_id, {"status": ReportStatus.FAILED, "error": error[:500], "completed_at": utcnow()}
        )


class PredictionStateStore:
    """Small key/value helper over ``worker_state`` for module-level settings (signal weights, accuracy)."""

    def __init__(self, db: AsyncDatabase[dict[str, Any]]) -> None:
        self.col = db[Collections.WORKER_STATE]

    async def get(self, key: str) -> dict[str, Any] | None:
        doc = await self.col.find_one({"key": key})
        value = doc.get("value") if doc else None
        return dict(value) if isinstance(value, dict) else None

    async def set(self, key: str, value: dict[str, Any], *, updated_by: str | None = None) -> None:
        await self.col.update_one(
            {"key": key},
            {
                "$set": {"value": value, "updated_at": utcnow(), "updated_by": updated_by},
                "$setOnInsert": {"key": key, "created_at": utcnow()},
            },
            upsert=True,
        )


__all__ = [
    "ACCURACY_STATE_KEY",
    "WEIGHTS_STATE_KEY",
    "MLModelRepository",
    "PredictionRepository",
    "PredictionStateStore",
    "ReportRepository",
    "TradeSignalRepository",
    "VolatilityRepository",
]
