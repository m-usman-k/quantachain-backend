"""Read-side service for Module 2 (the ingestion workers live in ``ingestion.py``)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationFailedError
from app.core.pagination import PageParams
from app.core.timeutils import SUPPORTED_INTERVALS, utcnow
from app.db.collections import Collections
from app.db.ingestion_logs import tail_ingestion_logs
from app.integrations.binance import BinanceClient
from app.integrations.coingecko import DEFAULT_COINGECKO_IDS
from app.modules.market.models import Asset, AssetStats, IngestionSource, NewsArticle, SocialPost
from app.modules.market.repository import (
    AssetRepository,
    AssetStatsRepository,
    CandleRepository,
    IngestionSourceRepository,
    MarketGlobalRepository,
    NewsRepository,
    SocialRepository,
    TickRepository,
)
from app.modules.market.schemas import (
    AssetDetail,
    AssetRow,
    CandleOut,
    IndicatorValues,
    IngestionOverview,
    IngestionSourceOut,
    LogLine,
    MarketDepthLevel,
    MarketDepthOut,
    MarketOverview,
    MarketPulse,
    SentimentBadge,
    TechnicalIndicatorsOut,
    Ticker,
)
from app.modules.market.seed import ASSET_NAMES
from app.modules.market.symbols import SymbolDetector
from app.modules.prediction.features import (
    atr,
    bollinger_pct_b,
    ema,
    log_returns,
    rolling_mean,
    rolling_std,
    rsi,
)

TABS = {"hot", "gainers", "losers", "volume", "new", "market_cap"}
STALE_HEARTBEAT = timedelta(seconds=75)


class MarketService:
    def __init__(self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.assets = AssetRepository(db)
        self.candles = CandleRepository(db)
        self.ticks = TickRepository(db)
        self.stats = AssetStatsRepository(db)
        self.global_repo = MarketGlobalRepository(db)
        self.news = NewsRepository(db)
        self.social = SocialRepository(db)
        self.sources = IngestionSourceRepository(db)

    # ---------------------------------------------------------- sentiment
    async def latest_sentiment(self, symbols: list[str] | None) -> dict[str | None, SentimentBadge]:
        """Latest Module 4 snapshot per symbol (``None`` key = global)."""
        match: dict[str, Any] = {"ts": {"$gte": utcnow() - timedelta(days=2)}}
        if symbols is None:
            match["meta.scope"] = "global"
        else:
            match["$or"] = [{"meta.scope": "global"}, {"meta.scope": "asset", "meta.symbol": {"$in": symbols}}]
        pipeline = [
            {"$match": match},
            {"$sort": {"ts": -1}},
            {
                "$group": {
                    "_id": {"scope": "$meta.scope", "symbol": "$meta.symbol"},
                    "index": {"$first": "$index"},
                    "label": {"$first": "$label"},
                    "score": {"$first": "$score"},
                    "ts": {"$first": "$ts"},
                }
            },
        ]
        cursor = await self.db[Collections.SENTIMENT_SNAPSHOTS].aggregate(pipeline)
        result: dict[str | None, SentimentBadge] = {}
        async for row in cursor:
            key = None if row["_id"].get("scope") == "global" else row["_id"].get("symbol")
            result[key] = SentimentBadge(
                index=row.get("index"), label=row.get("label"), score=row.get("score"), as_of=row.get("ts")
            )
        return result

    # ------------------------------------------------------------ overview
    async def overview(self, *, limit: int = 10) -> MarketOverview:
        global_doc = await self.global_repo.get_global()
        rows = await self.list_assets("market_cap", limit=limit)
        sentiment = await self.latest_sentiment(None)
        active = await self.sources.count(
            {"status": {"$in": ["connected", "active"]}, "heartbeat_at": {"$gte": utcnow() - STALE_HEARTBEAT}}
        )
        pulse = MarketPulse(
            fear_greed_value=global_doc.fear_greed_value if global_doc else None,
            fear_greed_label=global_doc.fear_greed_label if global_doc else None,
            total_market_cap_usd=global_doc.total_market_cap_usd if global_doc else None,
            total_volume_24h_usd=global_doc.total_volume_24h_usd if global_doc else None,
            btc_dominance_pct=global_doc.btc_dominance_pct if global_doc else None,
            eth_dominance_pct=global_doc.eth_dominance_pct if global_doc else None,
            market_cap_change_24h_pct=global_doc.market_cap_change_24h_pct if global_doc else None,
            active_cryptocurrencies=global_doc.active_cryptocurrencies if global_doc else None,
            updated_at=global_doc.updated_at if global_doc else None,
        )
        return MarketOverview(
            pulse=pulse,
            assets=rows,
            global_sentiment=sentiment.get(None, SentimentBadge()),
            ingestion_active=active > 0,
            generated_at=utcnow(),
        )

    async def list_assets(self, tab: str = "hot", *, limit: int = 50) -> list[AssetRow]:
        if tab not in TABS:
            raise ValidationFailedError(f"Unknown tab '{tab}'. Use one of {sorted(TABS)}")
        assets = await self.assets.tracked()
        by_symbol = {a.symbol: a for a in assets}
        stats = {s.symbol: s for s in await self.stats.find({"symbol": {"$in": list(by_symbol)}})}
        sentiment = await self.latest_sentiment(list(by_symbol))
        rows = [AssetRow.build(asset, stats.get(asset.symbol), sentiment.get(asset.symbol)) for asset in assets]

        def key_desc(attr: str):  # type: ignore[no-untyped-def]
            return lambda r: (getattr(r, attr) is None, -(getattr(r, attr) or 0))

        if tab == "gainers":
            rows.sort(key=key_desc("change_24h_pct"))
        elif tab == "losers":
            rows.sort(key=lambda r: (r.change_24h_pct is None, r.change_24h_pct or 0))
        elif tab == "volume":
            rows.sort(key=key_desc("volume_24h"))
        elif tab == "new":
            created = {a.symbol: a.created_at for a in assets}
            rows.sort(key=lambda r: created.get(r.symbol) or datetime.min, reverse=True)
        elif tab == "market_cap":
            rows.sort(key=lambda r: (r.market_cap is None, -(r.market_cap or 0), r.rank or 9999))
        else:  # hot: volume relative to market cap, falling back to rank
            rows.sort(
                key=lambda r: (-((r.volume_24h or 0) / (r.market_cap or 1)) if r.volume_24h else 0, r.rank or 9999)
            )
        return rows[:limit]

    async def asset_detail(self, symbol: str) -> AssetDetail:
        asset = await self.assets.get_by_symbol(symbol)
        if asset is None:
            raise NotFoundError(f"Asset {symbol.upper()} is not tracked")
        stats = await self.stats.get_by_symbol(asset.symbol)
        sentiment = await self.latest_sentiment([asset.symbol])
        latest = await self.candles.latest(asset.symbol, "1h") or await self.candles.latest(asset.symbol, "1d")
        return AssetDetail(
            asset=asset,
            row=AssetRow.build(asset, stats, sentiment.get(asset.symbol)),
            latest_candle=CandleOut.from_candle(latest) if latest else None,
            ticker=self._ticker(asset.symbol, stats),
        )

    async def ticker(self, symbol: str) -> Ticker:
        stats = await self.stats.get_by_symbol(symbol)
        if stats is None and await self.assets.get_by_symbol(symbol) is None:
            raise NotFoundError(f"Asset {symbol.upper()} is not tracked")
        return self._ticker(symbol.upper(), stats)

    @staticmethod
    def _ticker(symbol: str, stats: AssetStats | None) -> Ticker:
        return Ticker(
            symbol=symbol,
            price=stats.price if stats else None,
            change_24h_pct=stats.change_24h_pct if stats else None,
            volume_24h=stats.volume_24h if stats else None,
            high_24h=stats.high_24h if stats else None,
            low_24h=stats.low_24h if stats else None,
            source=stats.price_source if stats else None,
            updated_at=stats.price_updated_at if stats else None,
        )

    async def latest_price(self, symbol: str) -> float | None:
        stats = await self.stats.get_by_symbol(symbol)
        if stats and stats.price is not None:
            return stats.price
        for interval in ("1m", "1h", "1d"):
            candle = await self.candles.latest(symbol, interval)
            if candle:
                return candle.close
        return None

    async def candle_series(
        self, symbol: str, interval: str, *, limit: int, start: datetime | None, end: datetime | None
    ) -> list[CandleOut]:
        if interval not in SUPPORTED_INTERVALS:
            raise ValidationFailedError(f"Unsupported interval '{interval}'. Use one of {list(SUPPORTED_INTERVALS)}")
        candles = await self.candles.range(symbol, interval, start=start, end=end, limit=limit)
        return [CandleOut.from_candle(c) for c in candles]

    async def compute_indicators(
        self, symbol: str, interval: str = "1h", *, limit: int = 100
    ) -> TechnicalIndicatorsOut:
        if interval not in SUPPORTED_INTERVALS:
            raise ValidationFailedError(f"Unsupported interval '{interval}'. Use one of {list(SUPPORTED_INTERVALS)}")
        candles = await self.candles.range(symbol, interval, limit=max(limit, 50))
        if not candles:
            raise NotFoundError(f"No candles found for asset {symbol.upper()}")

        sorted_candles = sorted(candles, key=lambda c: c.open_time)
        closes = np.array([c.close for c in sorted_candles], dtype=float)
        highs = np.array([c.high for c in sorted_candles], dtype=float)
        lows = np.array([c.low for c in sorted_candles], dtype=float)
        n = closes.size
        last_close = float(closes[-1])

        sma20_arr = rolling_mean(closes, 20) if n >= 20 else np.full(n, np.nan)
        sma50_arr = rolling_mean(closes, 50) if n >= 50 else np.full(n, np.nan)
        sma200_arr = rolling_mean(closes, 200) if n >= 200 else np.full(n, np.nan)

        ema12_arr = ema(closes, 12) if n >= 12 else np.full(n, np.nan)
        ema26_arr = ema(closes, 26) if n >= 26 else np.full(n, np.nan)
        ema50_arr = ema(closes, 50) if n >= 50 else np.full(n, np.nan)

        rsi_arr = rsi(closes, 14) if n > 14 else np.full(n, np.nan)

        if n >= 26:
            macd_line_arr = ema12_arr - ema26_arr
            macd_sig_arr = ema(macd_line_arr, 9)
            macd_hist_arr = macd_line_arr - macd_sig_arr
        else:
            macd_line_arr = np.full(n, np.nan)
            macd_sig_arr = np.full(n, np.nan)
            macd_hist_arr = np.full(n, np.nan)

        if n >= 20:
            sd20 = rolling_std(closes, 20)
            bb_mid = sma20_arr[-1]
            bb_upper = bb_mid + 2.0 * sd20[-1] if not np.isnan(sd20[-1]) else None
            bb_lower = bb_mid - 2.0 * sd20[-1] if not np.isnan(sd20[-1]) else None
            pct_b_arr = bollinger_pct_b(closes, 20)
            bb_pct_b_val = float(pct_b_arr[-1]) if not np.isnan(pct_b_arr[-1]) else None
        else:
            bb_mid = None
            bb_upper = None
            bb_lower = None
            bb_pct_b_val = None

        atr_arr = atr(highs, lows, closes, 14) if n > 14 else np.full(n, np.nan)

        rets = log_returns(closes)
        valid_rets = rets[~np.isnan(rets)]
        if valid_rets.size >= 12:
            vol_24h = float(np.std(valid_rets[-24:], ddof=1) * np.sqrt(24) * 100)
        else:
            vol_24h = None

        def _val(arr: np.ndarray) -> float | None:
            v = arr[-1] if arr.size > 0 else np.nan
            return round(float(v), 4) if not np.isnan(v) else None

        sma_20_val = _val(sma20_arr)
        sma_50_val = _val(sma50_arr)
        sma_200_val = _val(sma200_arr)
        ema_12_val = _val(ema12_arr)
        ema_26_val = _val(ema26_arr)
        ema_50_val = _val(ema50_arr)
        rsi_14_val = _val(rsi_arr)
        macd_line_val = _val(macd_line_arr)
        macd_sig_val = _val(macd_sig_arr)
        macd_hist_val = _val(macd_hist_arr)
        atr_14_val = _val(atr_arr)

        bullish_score = 0.0
        bearish_score = 0.0
        reasons: list[str] = []

        if rsi_14_val is not None:
            if rsi_14_val > 70:
                reasons.append(f"RSI(14) is overbought at {rsi_14_val:.1f}")
                bearish_score += 1.0
            elif rsi_14_val < 30:
                reasons.append(f"RSI(14) is oversold at {rsi_14_val:.1f}")
                bullish_score += 1.0
            elif rsi_14_val >= 50:
                reasons.append(f"RSI(14) is in positive territory at {rsi_14_val:.1f}")
                bullish_score += 0.5
            else:
                reasons.append(f"RSI(14) is below midline at {rsi_14_val:.1f}")
                bearish_score += 0.5

        if macd_hist_val is not None:
            if macd_hist_val > 0:
                reasons.append("MACD momentum is positive")
                bullish_score += 1.0
            else:
                reasons.append("MACD momentum is negative")
                bearish_score += 1.0

        if sma_50_val is not None:
            if last_close > sma_50_val:
                reasons.append(f"Price (${last_close:,.2f}) trades above 50-period SMA (${sma_50_val:,.2f})")
                bullish_score += 1.0
            else:
                reasons.append(f"Price (${last_close:,.2f}) trades below 50-period SMA (${sma_50_val:,.2f})")
                bearish_score += 1.0

        if bullish_score >= 2.0 and bullish_score > bearish_score:
            summary = "bullish"
        elif bearish_score >= 2.0 and bearish_score > bullish_score:
            summary = "bearish"
        else:
            summary = "neutral"

        indicators = IndicatorValues(
            close=last_close,
            sma_20=sma_20_val,
            sma_50=sma_50_val,
            sma_200=sma_200_val,
            ema_12=ema_12_val,
            ema_26=ema_26_val,
            ema_50=ema_50_val,
            rsi_14=rsi_14_val,
            macd_line=macd_line_val,
            macd_signal=macd_sig_val,
            macd_hist=macd_hist_val,
            bollinger_upper=round(bb_upper, 4) if bb_upper is not None else None,
            bollinger_middle=round(bb_mid, 4) if bb_mid is not None else None,
            bollinger_lower=round(bb_lower, 4) if bb_lower is not None else None,
            bollinger_pct_b=round(bb_pct_b_val, 4) if bb_pct_b_val is not None else None,
            atr_14=atr_14_val,
            volatility_24h_pct=round(vol_24h, 2) if vol_24h is not None else None,
        )

        return TechnicalIndicatorsOut(
            symbol=symbol.upper(),
            interval=interval,
            candle_count=n,
            indicators=indicators,
            signal_summary=summary,
            reasons=reasons,
            as_of=sorted_candles[-1].open_time,
        )

    async def market_depth(self, symbol: str, *, limit: int = 20) -> MarketDepthOut:
        sym = symbol.upper()
        bids: list[MarketDepthLevel] = []
        asks: list[MarketDepthLevel] = []
        source = "live"
        try:
            async with BinanceClient(self.settings) as client:
                book = await client.fetch_order_book(sym, limit=limit)
            bids = [MarketDepthLevel(price=b[0], amount=b[1]) for b in book["bids"]]
            asks = [MarketDepthLevel(price=a[0], amount=a[1]) for a in book["asks"]]
        except Exception:
            source = "synthetic"
            price = await self.latest_price(sym) or 50000.0
            for i in range(1, limit + 1):
                spread_step = 0.0005 * i
                bid_p = round(price * (1.0 - spread_step), 2)
                ask_p = round(price * (1.0 + spread_step), 2)
                amt = round(0.5 * (1 + 0.1 * i), 4)
                bids.append(MarketDepthLevel(price=bid_p, amount=amt))
                asks.append(MarketDepthLevel(price=ask_p, amount=amt))

        best_bid = bids[0].price if bids else None
        best_ask = asks[0].price if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
        spread_pct = round(spread / mid * 100, 6) if spread is not None and mid else None

        return MarketDepthOut(
            symbol=sym,
            pair=f"{sym}/{self.settings.quote_asset}",
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=round(mid, 8) if mid is not None else None,
            spread=round(spread, 8) if spread is not None else None,
            spread_pct=spread_pct,
            source=source,
            updated_at=utcnow(),
        )

    # ------------------------------------------------------------- content
    async def list_news(
        self,
        params: PageParams,
        *,
        symbol: str | None = None,
        query: str | None = None,
        source: str | None = None,
        analyzed_only: bool = False,
        geopolitical_only: bool = False,
    ) -> tuple[list[NewsArticle], int]:
        filters: dict[str, Any] = {}
        if symbol:
            filters["symbols"] = symbol.upper()
        if source:
            filters["source_name"] = {"$regex": source, "$options": "i"}
        if analyzed_only:
            filters["analysis.status"] = "done"
        if geopolitical_only:
            filters["analysis.is_geopolitical"] = True
        if query:
            filters["$text"] = {"$search": query}
        return await self.news.paginate(filters, params, sort=[("published_at", DESCENDING)])

    async def get_article(self, article_id: str) -> NewsArticle:
        return await self.news.get_or_raise(article_id)

    async def list_social(
        self,
        params: PageParams,
        *,
        symbol: str | None = None,
        platform: str | None = None,
        community: str | None = None,
    ) -> tuple[list[SocialPost], int]:
        filters: dict[str, Any] = {}
        if symbol:
            filters["symbols"] = symbol.upper()
        if platform:
            filters["platform"] = platform
        if community:
            filters["community"] = community
        return await self.social.paginate(filters, params, sort=[("posted_at", DESCENDING)])

    # ---------------------------------------------------------- ingestion
    async def ingestion_overview(self) -> IngestionOverview:
        sources = await self.sources.find({}, sort=[("label", 1)])
        now = utcnow()
        out = [IngestionSourceOut.from_model(s, stale=self._is_stale(s, now)) for s in sources]
        live = [s for s in sources if not self._is_stale(s, now)]
        latencies = [s.latency_ms for s in live if s.latency_ms is not None]
        return IngestionOverview(
            sources=out,
            events_per_minute=round(sum(s.events_per_minute for s in live), 1),
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
            worker_online=bool(live),
            ticks_last_hour=await self.ticks.count_since(now - timedelta(hours=1)),
            generated_at=now,
        )

    @staticmethod
    def _is_stale(source: IngestionSource, now: datetime) -> bool:
        return source.heartbeat_at is None or now - source.heartbeat_at > STALE_HEARTBEAT

    async def set_source_enabled(self, name: str, enabled: bool) -> IngestionSourceOut:
        source = await self.sources.set_enabled(name, enabled)
        if source is None:
            raise NotFoundError("Ingestion source not found")
        return IngestionSourceOut.from_model(source, stale=self._is_stale(source, utcnow()))

    async def recent_logs(self, limit: int = 100) -> list[LogLine]:
        rows = await tail_ingestion_logs(self.db, limit=limit)
        return [
            LogLine(
                id=str(r["_id"]),
                ts=r["ts"],
                source=r.get("source", ""),
                level=r.get("level", "info"),
                message=r.get("message", ""),
            )
            for r in rows
        ]

    # ---------------------------------------------------------- management
    async def add_asset(
        self, symbol: str, *, name: str | None, coingecko_id: str | None, contracts: dict[str, str]
    ) -> Asset:
        symbol = symbol.upper()
        existing = await self.assets.get_by_symbol(symbol)
        if existing and existing.is_tracked:
            raise ConflictError(f"{symbol} is already tracked")
        return await self.assets.upsert_asset(
            symbol,
            name=name or ASSET_NAMES.get(symbol, symbol),
            coingecko_id=coingecko_id or DEFAULT_COINGECKO_IDS.get(symbol),
            binance_symbol=f"{symbol}{self.settings.quote_asset}",
            contracts=contracts or None,
            is_tracked=True,
        )

    async def remove_asset(self, symbol: str) -> None:
        asset = await self.assets.get_by_symbol(symbol)
        if asset is None:
            raise NotFoundError(f"Asset {symbol.upper()} is not tracked")
        await self.assets.update(asset.id or "", {"is_tracked": False})

    async def symbol_detector(self) -> SymbolDetector:
        return SymbolDetector(await self.assets.tracked())


__all__ = ["TABS", "MarketService"]
