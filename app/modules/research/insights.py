"""Monthly / quarterly macro insights aggregated from every module (FE-4)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.integrations.llm import LLMClient
from app.modules.research.models import InsightPeriod, MarketInsight
from app.modules.research.repository import MarketInsightRepository

logger = structlog.get_logger(__name__)


def period_bounds(period: InsightPeriod, reference: datetime) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` of the period containing ``reference`` (UTC, end exclusive)."""
    ref = reference.astimezone(UTC)
    if period == InsightPeriod.MONTHLY:
        start = ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=32)).replace(day=1)
    else:
        quarter_month = 3 * ((ref.month - 1) // 3) + 1
        start = ref.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (start + timedelta(days=93)).replace(day=1)
    return start, end


async def _returns(
    db: AsyncDatabase[dict[str, Any]], symbols: list[str], start: datetime, end: datetime
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        rows = (
            await db[Collections.CANDLES]
            .find(
                {"symbol": symbol, "interval": "1d", "open_time": {"$gte": start, "$lt": end}},
                {"open_time": 1, "open": 1, "close": 1, "high": 1, "low": 1},
            )
            .sort("open_time", 1)
            .to_list(length=400)
        )
        if len(rows) < 2:
            continue
        closes = [r["close"] for r in rows]
        daily = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0
        ]
        mean = sum(daily) / len(daily) if daily else 0.0
        vol = math.sqrt(sum((d - mean) ** 2 for d in daily) / len(daily)) * math.sqrt(365) if daily else 0.0
        peak, max_dd = closes[0], 0.0
        for close in closes:
            peak = max(peak, close)
            max_dd = min(max_dd, close / peak - 1)
        out[symbol] = {
            "return_pct": round((closes[-1] / rows[0]["open"] - 1) * 100, 2) if rows[0]["open"] else None,
            "high": max(r["high"] for r in rows),
            "low": min(r["low"] for r in rows),
            "annualised_volatility_pct": round(vol * 100, 1),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "days": len(rows),
        }
    return out


async def compute_insight(
    db: AsyncDatabase[dict[str, Any]],
    settings: Settings,
    period: InsightPeriod,
    reference: datetime | None = None,
    *,
    llm: LLMClient | None = None,
) -> MarketInsight:
    now = utcnow()
    start, end = period_bounds(period, reference or now)
    in_progress = end > now
    symbols = list(settings.tracked_symbols)
    performance = await _returns(db, symbols, start, min(end, now))
    ranked = sorted(performance.items(), key=lambda kv: kv[1]["return_pct"] or -1e9, reverse=True)

    sentiment_rows = (
        await db[Collections.SENTIMENT_SNAPSHOTS]
        .find({"meta.scope": "global", "ts": {"$gte": start, "$lt": end}}, {"index": 1})
        .to_list(length=5000)
    )
    sentiment_values = [r["index"] for r in sentiment_rows if r.get("index") is not None]
    avg_sentiment = round(sum(sentiment_values) / len(sentiment_values), 1) if sentiment_values else None

    whale_pipeline = [
        {"$match": {"block_time": {"$gte": start, "$lt": end}}},
        {
            "$group": {
                "_id": None,
                "volume": {"$sum": "$amount_usd"},
                "count": {"$sum": 1},
                "inflow": {"$sum": {"$cond": [{"$eq": ["$flow", "exchange_inflow"]}, "$amount_usd", 0]}},
                "outflow": {"$sum": {"$cond": [{"$eq": ["$flow", "exchange_outflow"]}, "$amount_usd", 0]}},
            }
        },
    ]
    whale_cursor = await db[Collections.WHALE_TRANSFERS].aggregate(whale_pipeline)
    whale_rows = await whale_cursor.to_list(length=1)
    whales = whale_rows[0] if whale_rows else {"volume": 0, "count": 0, "inflow": 0, "outflow": 0}

    alerts_cursor = await db[Collections.FRAUD_ALERTS].aggregate(
        [
            {"$match": {"created_at": {"$gte": start, "$lt": end}}},
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
        ]
    )
    alerts = {row["_id"]: row["count"] async for row in alerts_cursor}
    news_count = await db[Collections.NEWS_ARTICLES].count_documents({"published_at": {"$gte": start, "$lt": end}})
    topics_cursor = await db[Collections.NEWS_ARTICLES].aggregate(
        [
            {"$match": {"published_at": {"$gte": start, "$lt": end}, "analysis.topics.0": {"$exists": True}}},
            {"$unwind": "$analysis.topics"},
            {"$group": {"_id": "$analysis.topics", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 6},
        ]
    )
    topics = [row["_id"] async for row in topics_cursor]
    predictions_cursor = await db[Collections.PREDICTIONS].aggregate(
        [
            {"$match": {"created_at": {"$gte": start, "$lt": end}, "evaluated": True}},
            {"$group": {"_id": None, "n": {"$sum": 1}, "hits": {"$sum": {"$cond": ["$direction_hit", 1, 0]}}}},
        ]
    )
    prediction_rows = await predictions_cursor.to_list(length=1)
    accuracy = (
        round(100 * prediction_rows[0]["hits"] / prediction_rows[0]["n"], 1)
        if prediction_rows and prediction_rows[0]["n"]
        else None
    )

    metrics = {
        "btc_return_pct": performance.get("BTC", {}).get("return_pct"),
        "eth_return_pct": performance.get("ETH", {}).get("return_pct"),
        "avg_sentiment_index": avg_sentiment,
        "whale_volume_usd": round(whales.get("volume", 0) or 0, 2),
        "whale_transfers": whales.get("count", 0),
        "exchange_netflow_usd": round((whales.get("inflow", 0) or 0) - (whales.get("outflow", 0) or 0), 2),
        "fraud_alerts": alerts,
        "news_articles": news_count,
        "top_topics": topics,
        "forecast_directional_accuracy_pct": accuracy,
        "assets_covered": len(performance),
    }
    performers = {
        "best": [{"symbol": s, **v} for s, v in ranked[:3]],
        "worst": [{"symbol": s, **v} for s, v in ranked[-3:][::-1]] if len(ranked) > 3 else [],
        "table": [{"symbol": s, **v} for s, v in ranked],
    }
    label = (
        start.strftime("%B %Y") if period == InsightPeriod.MONTHLY else f"Q{(start.month - 1) // 3 + 1} {start.year}"
    )
    title = f"{'Monthly' if period == InsightPeriod.MONTHLY else 'Quarterly'} Market Insight - {label}" + (
        " (in progress)" if in_progress else ""
    )
    highlights = _highlights(metrics, performers, topics)
    summary, generated_by = await _narrative(llm, title, metrics, performers, highlights)

    insight = MarketInsight(
        period=period,
        period_start=start,
        period_end=end,
        title=title,
        summary=summary,
        highlights=highlights,
        metrics=metrics,
        performers=performers,
        symbols=symbols,
        generated_by=generated_by,
        in_progress=in_progress,
    )
    repo = MarketInsightRepository(db)
    data = insight.to_mongo()
    data.pop("_id", None)
    created_at = data.pop("created_at")
    data["updated_at"] = utcnow()
    await repo.col.update_one(
        {"period": str(period), "period_start": start},
        {"$set": data, "$setOnInsert": {"created_at": created_at}},
        upsert=True,
    )
    stored = await repo.find_one({"period": str(period), "period_start": start})
    return stored or insight


def _highlights(metrics: dict[str, Any], performers: dict[str, Any], topics: list[str]) -> list[str]:
    out: list[str] = []
    if metrics.get("btc_return_pct") is not None:
        out.append(
            f"BTC {metrics['btc_return_pct']:+.1f}% and ETH {metrics.get('eth_return_pct') or 0:+.1f}% over the period"
        )
    if performers["best"]:
        best = performers["best"][0]
        out.append(f"Best performer: {best['symbol']} ({best['return_pct']:+.1f}%)")
    if performers["worst"]:
        worst = performers["worst"][0]
        out.append(f"Weakest: {worst['symbol']} ({worst['return_pct']:+.1f}%)")
    if metrics.get("avg_sentiment_index") is not None:
        out.append(f"Average sentiment index {metrics['avg_sentiment_index']}/100")
    if metrics.get("whale_transfers"):
        direction = "into" if metrics["exchange_netflow_usd"] > 0 else "out of"
        out.append(
            f"{metrics['whale_transfers']} whale transfers (${metrics['whale_volume_usd']:,.0f}), net ${abs(metrics['exchange_netflow_usd']):,.0f} {direction} exchanges"
        )
    total_alerts = sum(metrics.get("fraud_alerts", {}).values())
    if total_alerts:
        out.append(f"{total_alerts} fraud alerts ({metrics['fraud_alerts'].get('critical', 0)} critical)")
    if topics:
        out.append("Dominant narratives: " + ", ".join(topics[:4]))
    if metrics.get("forecast_directional_accuracy_pct") is not None:
        out.append(f"Forecast directional accuracy {metrics['forecast_directional_accuracy_pct']}%")
    return out


async def _narrative(
    llm: LLMClient | None, title: str, metrics: dict[str, Any], performers: dict[str, Any], highlights: list[str]
) -> tuple[str, str]:
    if llm is not None and llm.available:
        try:
            text = await llm.complete_text(
                system="You are a crypto market strategist writing a concise, neutral macro summary (120-180 words) for retail investors. Use only the supplied figures.",
                user=f"{title}\nMetrics: {metrics}\nPerformers: {performers}\nHighlights: {highlights}",
                max_tokens=400,
            )
            if text:
                return text, "gpt-4o"
        except ExternalServiceError as exc:
            logger.warning("insight_narrative_llm_failed", error=str(exc))
    if not highlights:
        return "Not enough data was collected in this period to draw conclusions yet.", "template"
    return " ".join(h.rstrip(".") + "." for h in highlights), "template"


__all__ = ["compute_insight", "period_bounds"]
