"""Module 2 persistence models: assets, candles, market stats, news, social, ingestion sources."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.db.models import Document


class Asset(Document):
    symbol: str
    name: str
    coingecko_id: str | None = None
    binance_symbol: str | None = None  # exchange pair, e.g. BTCUSDT
    image_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    contracts: dict[str, str] = Field(default_factory=dict)  # chain -> contract address
    decimals: int | None = None
    is_tracked: bool = True
    rank: int | None = None


class Candle(Document):
    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    quote_volume: float = 0.0
    trades: int = 0
    source: str = "binance"
    closed: bool = True


class AssetStats(Document):
    """Latest market snapshot for one asset (the Markets table row)."""

    symbol: str
    name: str | None = None
    price: float | None = None
    change_1h_pct: float | None = None
    change_24h_pct: float | None = None
    change_7d_pct: float | None = None
    market_cap: float | None = None
    volume_24h: float | None = None
    rank: int | None = None
    circulating_supply: float | None = None
    total_supply: float | None = None
    ath: float | None = None
    ath_change_pct: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    image_url: str | None = None
    price_source: str | None = None
    price_updated_at: datetime | None = None
    stats_updated_at: datetime | None = None


class MarketGlobal(Document):
    total_market_cap_usd: float | None = None
    total_volume_24h_usd: float | None = None
    btc_dominance_pct: float | None = None
    eth_dominance_pct: float | None = None
    market_cap_change_24h_pct: float | None = None
    active_cryptocurrencies: int | None = None
    markets: int | None = None
    fear_greed_value: int | None = None
    fear_greed_label: str | None = None
    fear_greed_updated_at: datetime | None = None


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class SentimentLabel(StrEnum):
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class Entity(BaseModel):
    text: str
    type: str  # TOKEN | PROTOCOL | EXCHANGE | REGULATOR | ORG | PERSON | COUNTRY | EVENT
    symbol: str | None = None


class TextAnalysis(BaseModel):
    """Filled in by Module 4 for every news article and social post."""

    status: AnalysisStatus = AnalysisStatus.PENDING
    sentiment_score: float | None = Field(default=None, ge=-1, le=1)
    sentiment_label: SentimentLabel | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    entities: list[Entity] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    impact_sectors: dict[str, float] = Field(default_factory=dict)  # sector -> impact (-1..1)
    is_geopolitical: bool = False
    regions: list[str] = Field(default_factory=list)  # ISO-3166 alpha-2 codes
    summary: str | None = None
    model: str | None = None
    analyzed_at: datetime | None = None
    error: str | None = None


class NewsArticle(Document):
    url: str
    url_hash: str
    title: str
    summary: str | None = None
    content: str | None = None
    source_name: str
    author: str | None = None
    image_url: str | None = None
    published_at: datetime
    fetched_at: datetime
    language: str = "en"
    origin: str  # newsapi | rss
    symbols: list[str] = Field(default_factory=list)
    analysis: TextAnalysis = Field(default_factory=TextAnalysis)


class SocialPost(Document):
    platform: str  # reddit | x | telegram
    external_id: str
    content_hash: str
    title: str
    text: str
    url: str
    author: str | None = None
    posted_at: datetime
    score: int = 0
    comments: int = 0
    community: str | None = None
    symbols: list[str] = Field(default_factory=list)
    analysis: TextAnalysis = Field(default_factory=TextAnalysis)
    extra: dict[str, Any] = Field(default_factory=dict)


class SourceStatus(StrEnum):
    CONNECTED = "connected"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"
    DISABLED = "disabled"


class IngestionSource(Document):
    """Status row for the admin Data Ingestion Hub."""

    name: str
    label: str
    kind: str  # websocket | rest | rss | rpc
    engine: str  # downstream consumer shown in the UI, e.g. "Market Sentiment"
    description: str = ""
    status: SourceStatus = SourceStatus.STOPPED
    enabled: bool = True
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    last_event_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    events_total: int = 0
    events_per_minute: float = 0.0
    latency_ms: float | None = None
    uptime_pct: float | None = None
    error_count: int = 0
    restarts: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AnalysisStatus",
    "Asset",
    "AssetStats",
    "Candle",
    "Entity",
    "IngestionSource",
    "MarketGlobal",
    "NewsArticle",
    "SentimentLabel",
    "SocialPost",
    "SourceStatus",
    "TextAnalysis",
]
