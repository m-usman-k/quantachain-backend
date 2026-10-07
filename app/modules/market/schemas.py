"""API schemas for Module 2."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.market.models import (
    AnalysisStatus,
    Asset,
    AssetStats,
    Candle,
    IngestionSource,
    NewsArticle,
    SentimentLabel,
    SocialPost,
    SourceStatus,
    TextAnalysis,
)


class SentimentBadge(BaseModel):
    index: float | None = Field(default=None, description="0 (extreme fear) .. 100 (extreme greed)")
    label: SentimentLabel | None = None
    score: float | None = None
    as_of: datetime | None = None


class AssetRow(BaseModel):
    symbol: str
    name: str
    image_url: str | None
    rank: int | None
    price: float | None
    change_1h_pct: float | None
    change_24h_pct: float | None
    change_7d_pct: float | None
    market_cap: float | None
    volume_24h: float | None
    high_24h: float | None
    low_24h: float | None
    price_updated_at: datetime | None
    sentiment: SentimentBadge

    @classmethod
    def build(cls, asset: Asset, stats: AssetStats | None, sentiment: SentimentBadge | None) -> AssetRow:
        return cls(
            symbol=asset.symbol,
            name=asset.name,
            image_url=(stats.image_url if stats else None) or asset.image_url,
            rank=(stats.rank if stats else None) or asset.rank,
            price=stats.price if stats else None,
            change_1h_pct=stats.change_1h_pct if stats else None,
            change_24h_pct=stats.change_24h_pct if stats else None,
            change_7d_pct=stats.change_7d_pct if stats else None,
            market_cap=stats.market_cap if stats else None,
            volume_24h=stats.volume_24h if stats else None,
            high_24h=stats.high_24h if stats else None,
            low_24h=stats.low_24h if stats else None,
            price_updated_at=stats.price_updated_at if stats else None,
            sentiment=sentiment or SentimentBadge(),
        )


class MarketPulse(BaseModel):
    fear_greed_value: int | None
    fear_greed_label: str | None
    total_market_cap_usd: float | None
    total_volume_24h_usd: float | None
    btc_dominance_pct: float | None
    eth_dominance_pct: float | None
    market_cap_change_24h_pct: float | None
    active_cryptocurrencies: int | None
    updated_at: datetime | None


class MarketOverview(BaseModel):
    pulse: MarketPulse
    assets: list[AssetRow]
    global_sentiment: SentimentBadge
    ingestion_active: bool
    generated_at: datetime


class Ticker(BaseModel):
    symbol: str
    price: float | None
    change_24h_pct: float | None
    volume_24h: float | None
    high_24h: float | None
    low_24h: float | None
    source: str | None
    updated_at: datetime | None


class CandleOut(BaseModel):
    t: datetime = Field(description="Open time")
    o: float
    h: float
    l: float  # noqa: E741 - conventional OHLC key
    c: float
    v: float
    qv: float = Field(description="Quote volume")
    n: int = Field(description="Trades")
    closed: bool

    @classmethod
    def from_candle(cls, candle: Candle) -> CandleOut:
        return cls(
            t=candle.open_time,
            o=candle.open,
            h=candle.high,
            l=candle.low,
            c=candle.close,
            v=candle.volume,
            qv=candle.quote_volume,
            n=candle.trades,
            closed=candle.closed,
        )


class CandleSeries(BaseModel):
    symbol: str
    interval: str
    candles: list[CandleOut]
    count: int


class AssetDetail(BaseModel):
    asset: Asset
    row: AssetRow
    latest_candle: CandleOut | None
    ticker: Ticker


class AnalysisOut(BaseModel):
    status: AnalysisStatus
    sentiment_score: float | None
    sentiment_label: SentimentLabel | None
    confidence: float | None
    entities: list[dict[str, Any]]
    topics: list[str]
    impact_sectors: dict[str, float]
    is_geopolitical: bool
    regions: list[str]
    summary: str | None
    analyzed_at: datetime | None

    @classmethod
    def from_analysis(cls, a: TextAnalysis) -> AnalysisOut:
        return cls(
            status=a.status,
            sentiment_score=a.sentiment_score,
            sentiment_label=a.sentiment_label,
            confidence=a.confidence,
            entities=[e.model_dump() for e in a.entities],
            topics=a.topics,
            impact_sectors=a.impact_sectors,
            is_geopolitical=a.is_geopolitical,
            regions=a.regions,
            summary=a.summary,
            analyzed_at=a.analyzed_at,
        )


class NewsOut(BaseModel):
    id: str
    title: str
    url: str
    summary: str | None
    source_name: str
    author: str | None
    image_url: str | None
    published_at: datetime
    origin: str
    symbols: list[str]
    analysis: AnalysisOut

    @classmethod
    def from_model(cls, a: NewsArticle) -> NewsOut:
        return cls(
            id=a.id or "",
            title=a.title,
            url=a.url,
            summary=a.summary,
            source_name=a.source_name,
            author=a.author,
            image_url=a.image_url,
            published_at=a.published_at,
            origin=a.origin,
            symbols=a.symbols,
            analysis=AnalysisOut.from_analysis(a.analysis),
        )


class SocialOut(BaseModel):
    id: str
    platform: str
    title: str
    text: str
    url: str
    author: str | None
    community: str | None
    posted_at: datetime
    score: int
    comments: int
    symbols: list[str]
    analysis: AnalysisOut

    @classmethod
    def from_model(cls, p: SocialPost) -> SocialOut:
        return cls(
            id=p.id or "",
            platform=p.platform,
            title=p.title,
            text=p.text[:500],
            url=p.url,
            author=p.author,
            community=p.community,
            posted_at=p.posted_at,
            score=p.score,
            comments=p.comments,
            symbols=p.symbols,
            analysis=AnalysisOut.from_analysis(p.analysis),
        )


class IngestionSourceOut(BaseModel):
    name: str
    label: str
    kind: str
    engine: str
    description: str
    status: SourceStatus
    enabled: bool
    started_at: datetime | None
    heartbeat_at: datetime | None
    last_event_at: datetime | None
    last_error: str | None
    last_error_at: datetime | None
    events_total: int
    events_per_minute: float
    latency_ms: float | None
    uptime_pct: float | None
    error_count: int
    restarts: int
    stale: bool = Field(description="No heartbeat from the worker for over a minute")

    @classmethod
    def from_model(cls, s: IngestionSource, *, stale: bool) -> IngestionSourceOut:
        return cls(**s.model_dump(exclude={"id", "created_at", "updated_at", "extra"}), stale=stale)


class IngestionOverview(BaseModel):
    sources: list[IngestionSourceOut]
    events_per_minute: float
    avg_latency_ms: float | None
    worker_online: bool
    ticks_last_hour: int
    generated_at: datetime


class LogLine(BaseModel):
    id: str
    ts: datetime
    source: str
    level: str
    message: str


class AddAssetRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, pattern=r"^[A-Za-z0-9]+$")
    name: str | None = Field(default=None, max_length=80)
    coingecko_id: str | None = Field(default=None, max_length=80)
    contracts: dict[str, str] = Field(default_factory=dict)


class ToggleRequest(BaseModel):
    enabled: bool


__all__ = [name for name in globals() if name[0].isupper()]
