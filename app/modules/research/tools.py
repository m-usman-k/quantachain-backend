"""Data tools the research assistant can call (FE-3).

Each tool reads from the shared collections using the cross-module contracts in
``docs/ARCHITECTURE.md`` so the assistant keeps working even when a module is
disabled. Tools return JSON-serialisable dicts that double as chart data.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.integrations.evm import is_address
from app.modules.market.service import MarketService

ToolHandler = Callable[[AsyncDatabase[dict[str, Any]], Settings, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _symbol_param(required: bool = True) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "Asset ticker such as BTC or ETH"}},
    }
    if required:
        schema["required"] = ["symbol"]
    return schema


def _serialise(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items() if k != "_id"}
    if isinstance(value, list):
        return [_serialise(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


# ------------------------------------------------------------------ tools
async def get_market_overview(
    db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]
) -> dict[str, Any]:
    overview = await MarketService(db, settings).overview(limit=int(args.get("limit", 10)))
    return _serialise(overview.model_dump())


async def get_price(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    symbol = str(args.get("symbol", "BTC")).upper()
    service = MarketService(db, settings)
    ticker = await service.ticker(symbol)
    return _serialise(ticker.model_dump())


async def get_candles_summary(
    db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]
) -> dict[str, Any]:
    symbol = str(args.get("symbol", "BTC")).upper()
    interval = str(args.get("interval", "1h"))
    lookback = int(args.get("lookback", 168))
    rows = (
        await db[Collections.CANDLES]
        .find(
            {"symbol": symbol, "interval": interval},
            {"open_time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "quote_volume": 1},
        )
        .sort("open_time", -1)
        .limit(lookback)
        .to_list(length=lookback)
    )
    rows.reverse()
    if not rows:
        return {"symbol": symbol, "interval": interval, "available": False}
    closes = [r["close"] for r in rows]
    highs = [r["high"] for r in rows]
    lows = [r["low"] for r in rows]
    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]
    gains = [r for r in returns[-14:] if r > 0]
    losses = [-r for r in returns[-14:] if r < 0]
    avg_gain = sum(gains) / 14 if gains else 0.0
    avg_loss = sum(losses) / 14 if losses else 0.0
    rsi = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    mean = sum(returns) / len(returns) if returns else 0.0
    vol = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns)) if returns else 0.0
    return {
        "symbol": symbol,
        "interval": interval,
        "available": True,
        "candles": len(rows),
        "from": rows[0]["open_time"].isoformat(),
        "to": rows[-1]["open_time"].isoformat(),
        "first_close": closes[0],
        "last_close": closes[-1],
        "change_pct": round((closes[-1] / closes[0] - 1) * 100, 3) if closes[0] else None,
        "high": max(highs),
        "low": min(lows),
        "volume_quote": round(sum(r.get("quote_volume") or 0 for r in rows), 2),
        "rsi_14": round(rsi, 1),
        "volatility_per_bar_pct": round(vol * 100, 3),
        "closes_sample": closes[-24:],
    }


async def get_sentiment(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    symbol = args.get("symbol")
    match: dict[str, Any] = (
        {"meta.scope": "asset", "meta.symbol": str(symbol).upper()} if symbol else {"meta.scope": "global"}
    )
    latest = await db[Collections.SENTIMENT_SNAPSHOTS].find_one(match, sort=[("ts", -1)])
    if latest is None:
        return {
            "symbol": symbol,
            "available": False,
            "note": "No sentiment snapshot yet; the sentiment engine may still be warming up.",
        }
    day_ago = await db[Collections.SENTIMENT_SNAPSHOTS].find_one(
        {**match, "ts": {"$lte": latest["ts"] - timedelta(hours=24)}}, sort=[("ts", -1)]
    )
    return _serialise(
        {
            "symbol": symbol,
            "available": True,
            "index": latest.get("index"),
            "label": latest.get("label"),
            "score": latest.get("score"),
            "sample_size": latest.get("sample_size"),
            "breakdown": latest.get("breakdown"),
            "change_24h": (latest.get("index") - day_ago.get("index"))
            if day_ago and latest.get("index") is not None and day_ago.get("index") is not None
            else None,
            "as_of": latest.get("ts"),
        }
    )


async def get_news(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    symbol = args.get("symbol")
    limit = min(int(args.get("limit", 8)), 20)
    query: dict[str, Any] = {}
    if symbol:
        query["symbols"] = str(symbol).upper()
    if args.get("geopolitical_only"):
        query["analysis.is_geopolitical"] = True
    rows = (
        await db[Collections.NEWS_ARTICLES]
        .find(
            query,
            {
                "title": 1,
                "source_name": 1,
                "published_at": 1,
                "url": 1,
                "symbols": 1,
                "analysis.sentiment_label": 1,
                "analysis.sentiment_score": 1,
                "analysis.summary": 1,
            },
        )
        .sort("published_at", -1)
        .limit(limit)
        .to_list(length=limit)
    )
    return {
        "symbol": symbol,
        "count": len(rows),
        "articles": [
            {
                "title": r["title"],
                "source": r.get("source_name"),
                "published_at": r["published_at"].isoformat(),
                "url": r.get("url"),
                "symbols": r.get("symbols", []),
                "sentiment": (r.get("analysis") or {}).get("sentiment_label"),
                "score": (r.get("analysis") or {}).get("sentiment_score"),
                "summary": (r.get("analysis") or {}).get("summary"),
            }
            for r in rows
        ],
    }


async def get_whale_activity(
    db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]
) -> dict[str, Any]:
    symbol = args.get("symbol")
    hours = min(int(args.get("hours", 24)), 24 * 30)
    since = utcnow() - timedelta(hours=hours)
    match: dict[str, Any] = {"block_time": {"$gte": since}}
    if symbol:
        match["symbol"] = str(symbol).upper()
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": "$flow",
                "volume_usd": {"$sum": "$amount_usd"},
                "count": {"$sum": 1},
            }
        },
    ]
    cursor = await db[Collections.WHALE_TRANSFERS].aggregate(pipeline)
    by_flow = {row["_id"]: {"volume_usd": round(row["volume_usd"], 2), "count": row["count"]} async for row in cursor}
    largest = (
        await db[Collections.WHALE_TRANSFERS]
        .find(
            match,
            {
                "symbol": 1,
                "amount": 1,
                "amount_usd": 1,
                "from_label": 1,
                "to_label": 1,
                "flow": 1,
                "risk_level": 1,
                "block_time": 1,
                "tx_hash": 1,
                "commentary": 1,
            },
        )
        .sort("amount_usd", -1)
        .limit(5)
        .to_list(length=5)
    )
    inflow = by_flow.get("exchange_inflow", {}).get("volume_usd", 0.0)
    outflow = by_flow.get("exchange_outflow", {}).get("volume_usd", 0.0)
    return _serialise(
        {
            "symbol": symbol,
            "hours": hours,
            "transfers": sum(v["count"] for v in by_flow.values()),
            "volume_usd": round(sum(v["volume_usd"] for v in by_flow.values()), 2),
            "exchange_inflow_usd": inflow,
            "exchange_outflow_usd": outflow,
            "exchange_netflow_usd": round(inflow - outflow, 2),
            "by_flow": by_flow,
            "largest": largest,
            "available": bool(by_flow),
        }
    )


async def get_prediction(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    symbol = str(args.get("symbol", "BTC")).upper()
    rows = (
        await db[Collections.PREDICTIONS].find({"symbol": symbol}).sort("created_at", -1).limit(12).to_list(length=12)
    )
    latest: dict[str, Any] = {}
    for row in rows:
        latest.setdefault(row.get("horizon"), row)
    if not latest:
        return {"symbol": symbol, "available": False, "note": "No forecast yet for this asset."}
    return _serialise(
        {
            "symbol": symbol,
            "available": True,
            "horizons": [
                {
                    "horizon": h,
                    "target_price": r.get("target_price"),
                    "lower": r.get("lower"),
                    "upper": r.get("upper"),
                    "confidence": r.get("confidence"),
                    "direction": r.get("direction"),
                    "change_pct": r.get("change_pct"),
                    "model": r.get("model"),
                    "created_at": r.get("created_at"),
                    "target_time": r.get("target_time"),
                }
                for h, r in latest.items()
            ],
        }
    )


async def get_volatility(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    symbol = str(args.get("symbol", "BTC")).upper()
    row = await db[Collections.VOLATILITY_FORECASTS].find_one({"symbol": symbol}, sort=[("created_at", -1)])
    if row is None:
        return {"symbol": symbol, "available": False}
    return _serialise({"symbol": symbol, "available": True, **{k: v for k, v in row.items() if k not in ("_id",)}})


async def get_fraud_alerts(
    db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]
) -> dict[str, Any]:
    limit = min(int(args.get("limit", 10)), 50)
    query: dict[str, Any] = {}
    if args.get("severity"):
        query["severity"] = args["severity"]
    rows = await db[Collections.FRAUD_ALERTS].find(query).sort("created_at", -1).limit(limit).to_list(length=limit)
    return _serialise(
        {"count": len(rows), "alerts": [{k: v for k, v in r.items() if k not in ("_id", "evidence")} for r in rows]}
    )


async def check_wallet(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    address = str(args.get("address", "")).lower()
    if not is_address(address):
        return {"address": address, "valid": False, "note": "Not a valid EVM address"}
    listed = await db[Collections.WALLET_BLACKLIST].find_one({"address": address})
    profile = await db[Collections.WALLETS].find_one({"address": address})
    transfers = await db[Collections.WHALE_TRANSFERS].count_documents(
        {"$or": [{"from_address": address}, {"to_address": address}]}
    )
    return _serialise(
        {
            "address": address,
            "valid": True,
            "blacklisted": bool(listed),
            "blacklist": {k: v for k, v in (listed or {}).items() if k != "_id"},
            "profile": {k: v for k, v in (profile or {}).items() if k != "_id"},
            "whale_transfers_seen": transfers,
        }
    )


async def scan_contract(db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]) -> dict[str, Any]:
    """Delegates to Module 6 when it is installed."""
    try:
        from app.modules.fraud.service import FraudService  # type: ignore[import-not-found]
    except Exception:
        return {"available": False, "note": "The fraud scanner module is not available."}
    service = FraudService(db, settings)
    scan = getattr(service, "scan", None)
    if scan is None:
        return {"available": False, "note": "The fraud scanner does not expose a scan method."}
    try:
        result = await scan(
            address=args.get("address"), source=args.get("source"), chain=args.get("chain", "eth"), user_id=None
        )
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    data = result.model_dump() if hasattr(result, "model_dump") else result
    return _serialise({"available": True, **{k: v for k, v in dict(data).items() if k not in ("source_hash",)}})


async def get_trending_topics(
    db: AsyncDatabase[dict[str, Any]], settings: Settings, args: dict[str, Any]
) -> dict[str, Any]:
    since = utcnow() - timedelta(hours=int(args.get("hours", 24)))
    pipeline = [
        {"$match": {"published_at": {"$gte": since}, "analysis.topics.0": {"$exists": True}}},
        {"$unwind": "$analysis.topics"},
        {
            "$group": {
                "_id": "$analysis.topics",
                "count": {"$sum": 1},
                "sentiment": {"$avg": "$analysis.sentiment_score"},
            }
        },
        {"$sort": {"count": -1}},
        {"$limit": int(args.get("limit", 10))},
    ]
    cursor = await db[Collections.NEWS_ARTICLES].aggregate(pipeline)
    topics = [
        {
            "topic": row["_id"],
            "count": row["count"],
            "sentiment": round(row["sentiment"], 3) if row.get("sentiment") is not None else None,
        }
        async for row in cursor
    ]
    return {"hours": int(args.get("hours", 24)), "topics": topics}


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "get_market_overview",
            "Market pulse (fear & greed, dominance, total cap) and the top assets table.",
            {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}},
            get_market_overview,
        ),
        ToolSpec("get_price", "Latest price and 24h statistics for an asset.", _symbol_param(), get_price),
        ToolSpec(
            "get_candles_summary",
            "OHLCV summary over a lookback window: change, high/low, volume, RSI(14) and volatility.",
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "interval": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]},
                    "lookback": {"type": "integer", "minimum": 10, "maximum": 1000},
                },
                "required": ["symbol"],
            },
            get_candles_summary,
        ),
        ToolSpec(
            "get_sentiment",
            "Latest sentiment index (0-100), label and breakdown for an asset or the whole market when symbol is omitted.",
            _symbol_param(required=False),
            get_sentiment,
        ),
        ToolSpec(
            "get_news",
            "Recent analysed headlines, optionally filtered to an asset or geopolitical items.",
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "geopolitical_only": {"type": "boolean"},
                },
            },
            get_news,
        ),
        ToolSpec(
            "get_whale_activity",
            "Whale transfer volume, exchange inflow/outflow/netflow and the largest transfers in a time window.",
            {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "hours": {"type": "integer", "minimum": 1, "maximum": 720},
                },
            },
            get_whale_activity,
        ),
        ToolSpec(
            "get_prediction",
            "Latest AI price forecasts (1h/4h/24h) with confidence and intervals.",
            _symbol_param(),
            get_prediction,
        ),
        ToolSpec(
            "get_volatility", "Latest volatility forecast and regime for an asset.", _symbol_param(), get_volatility
        ),
        ToolSpec(
            "get_fraud_alerts",
            "Recent fraud / rug-pull / pump-and-dump alerts.",
            {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                },
            },
            get_fraud_alerts,
        ),
        ToolSpec(
            "check_wallet",
            "Check whether a wallet address is blacklisted and summarise its on-chain profile.",
            {"type": "object", "properties": {"address": {"type": "string"}}, "required": ["address"]},
            check_wallet,
        ),
        ToolSpec(
            "scan_contract",
            "Scan a smart contract (by address or Solidity source) for vulnerabilities and rug-pull patterns.",
            {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "source": {"type": "string"},
                    "chain": {"type": "string"},
                },
            },
            scan_contract,
        ),
        ToolSpec(
            "get_trending_topics",
            "Topics trending in analysed news over the last hours.",
            {"type": "object", "properties": {"hours": {"type": "integer"}, "limit": {"type": "integer"}}},
            get_trending_topics,
        ),
    ]
}


async def run_tool(
    db: AsyncDatabase[dict[str, Any]], settings: Settings, name: str, args: dict[str, Any]
) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        return {"error": f"Unknown tool {name}"}
    try:
        return await spec.handler(db, settings, args or {})
    except Exception as exc:  # tools must never break the conversation
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = ["TOOLS", "ToolSpec", "run_tool"]
