"""Aligned multi-metric series for the interactive charts (FE-1)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.exceptions import ValidationFailedError
from app.core.timeutils import SUPPORTED_INTERVALS, interval_seconds, parse_range, utcnow
from app.db.collections import Collections

CHART_METRICS = ("price", "volume", "sentiment", "netflow", "predictions", "whale_volume")


async def build_chart(
    db: AsyncDatabase[dict[str, Any]],
    symbol: str,
    *,
    metrics: list[str],
    interval: str,
    range_: str,
) -> dict[str, Any]:
    symbol = symbol.upper()
    if interval not in SUPPORTED_INTERVALS:
        raise ValidationFailedError(f"Unsupported interval '{interval}'")
    unknown = [m for m in metrics if m not in CHART_METRICS]
    if unknown:
        raise ValidationFailedError(f"Unknown metrics {unknown}; choose from {list(CHART_METRICS)}")
    now = utcnow()
    since = now - parse_range(range_)
    bucket = interval_seconds(interval)
    series: dict[str, list[dict[str, Any]]] = {}

    if "price" in metrics or "volume" in metrics:
        cursor = (
            db[Collections.CANDLES]
            .find(
                {"symbol": symbol, "interval": interval, "open_time": {"$gte": since}},
                {"open_time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "quote_volume": 1, "volume": 1},
            )
            .sort("open_time", 1)
            .limit(5000)
        )
        price, volume = [], []
        async for row in cursor:
            price.append(
                {"t": row["open_time"], "o": row["open"], "h": row["high"], "l": row["low"], "c": row["close"]}
            )
            volume.append({"t": row["open_time"], "v": row.get("quote_volume") or row.get("volume") or 0.0})
        if "price" in metrics:
            series["price"] = price
        if "volume" in metrics:
            series["volume"] = volume

    if "sentiment" in metrics:
        series["sentiment"] = await _bucketed(
            db,
            Collections.SENTIMENT_SNAPSHOTS,
            {"meta.scope": "asset", "meta.symbol": symbol, "ts": {"$gte": since}},
            value_expr={"$avg": "$index"},
            bucket=bucket,
            time_field="ts",
        )
        if not series["sentiment"]:
            series["sentiment"] = await _bucketed(
                db,
                Collections.SENTIMENT_SNAPSHOTS,
                {"meta.scope": "global", "ts": {"$gte": since}},
                value_expr={"$avg": "$index"},
                bucket=bucket,
                time_field="ts",
            )

    if "netflow" in metrics or "whale_volume" in metrics:
        pipeline = [
            {"$match": {"symbol": symbol, "block_time": {"$gte": since}}},
            {
                "$group": {
                    "_id": {"$dateTrunc": {"date": "$block_time", "unit": "second", "binSize": bucket}},
                    "inflow": {"$sum": {"$cond": [{"$eq": ["$flow", "exchange_inflow"]}, "$amount_usd", 0]}},
                    "outflow": {"$sum": {"$cond": [{"$eq": ["$flow", "exchange_outflow"]}, "$amount_usd", 0]}},
                    "volume": {"$sum": "$amount_usd"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = await db[Collections.WHALE_TRANSFERS].aggregate(pipeline)
        netflow, whale_volume = [], []
        async for row in cursor:
            netflow.append({"t": row["_id"], "v": round(row["inflow"] - row["outflow"], 2)})
            whale_volume.append({"t": row["_id"], "v": round(row["volume"], 2), "count": row["count"]})
        if "netflow" in metrics:
            series["netflow"] = netflow
        if "whale_volume" in metrics:
            series["whale_volume"] = whale_volume

    if "predictions" in metrics:
        cursor = (
            db[Collections.PREDICTIONS]
            .find(
                {"symbol": symbol, "created_at": {"$gte": since - timedelta(days=1)}},
                {
                    "horizon": 1,
                    "target_price": 1,
                    "lower": 1,
                    "upper": 1,
                    "confidence": 1,
                    "target_time": 1,
                    "created_at": 1,
                    "realized_price": 1,
                },
            )
            .sort("created_at", 1)
            .limit(2000)
        )
        series["predictions"] = [
            {
                "t": row.get("target_time"),
                "created_at": row.get("created_at"),
                "horizon": row.get("horizon"),
                "target": row.get("target_price"),
                "lower": row.get("lower"),
                "upper": row.get("upper"),
                "confidence": row.get("confidence"),
                "realized": row.get("realized_price"),
            }
            async for row in cursor
        ]

    return {"symbol": symbol, "interval": interval, "range": range_, "from": since, "to": now, "series": series}


async def _bucketed(
    db: AsyncDatabase[dict[str, Any]],
    collection: str,
    match: dict[str, Any],
    *,
    value_expr: dict[str, Any],
    bucket: int,
    time_field: str,
) -> list[dict[str, Any]]:
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": {"$dateTrunc": {"date": f"${time_field}", "unit": "second", "binSize": bucket}},
                "v": value_expr,
            }
        },
        {"$sort": {"_id": 1}},
    ]
    cursor = await db[collection].aggregate(pipeline)
    return [{"t": row["_id"], "v": round(row["v"], 3) if row["v"] is not None else None} async for row in cursor]


def _dt(value: datetime | None) -> datetime | None:
    return value


__all__ = ["CHART_METRICS", "build_chart"]
