"""Module 4 API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.market.models import Entity, SentimentLabel
from app.modules.sentiment.models import GeoCategory


class SentimentSnapshotOut(BaseModel):
    ts: datetime
    scope: str = "global"
    symbol: str | None = None
    score: float
    index: int
    label: SentimentLabel
    sample_size: int = 0
    breakdown: dict[str, float] = Field(default_factory=dict)


class SentimentOverviewOut(BaseModel):
    vibe_index: int
    score: float
    label: SentimentLabel
    fear_greed_index: int | None = None
    fear_greed_label: str | None = None
    change_24h: float = 0.0
    sample_size: int = 0
    asset_sentiment: dict[str, int] = Field(default_factory=dict)
    top_bullish: list[str] = Field(default_factory=list)
    top_bearish: list[str] = Field(default_factory=list)
    updated_at: datetime


class GeoEventOut(BaseModel):
    id: str
    title: str
    category: GeoCategory
    impacted_sectors: list[str] = Field(default_factory=list)
    sentiment_score: float
    sentiment_label: SentimentLabel
    summary: str
    regions: list[str] = Field(default_factory=list)
    region: str | None = None
    fingerprint: str = ""
    source_url: str | None = None
    source_name: str | None = None
    published_at: datetime
    observed_associations: list[dict[str, Any]] = Field(default_factory=list)


class SectorHeatmapItem(BaseModel):
    sector: str
    average_sentiment: float
    event_count: int
    status: str


class GeoHeatmapOut(BaseModel):
    sectors: list[SectorHeatmapItem] = Field(default_factory=list)
    regions: dict[str, float] = Field(default_factory=dict)
    categories: dict[str, int] = Field(default_factory=dict)
    total_events: int = 0
    updated_at: datetime


class CorrelationOut(BaseModel):
    symbol: str
    window_days: int
    correlation: float
    sample_count: int
    interpretation: str
    computed_at: datetime


class TextAnalyzeIn(BaseModel):
    text: str
    symbol: str | None = None


class TextAnalyzeOut(BaseModel):
    sentiment_score: float
    sentiment_label: SentimentLabel
    confidence: float
    entities: list[Entity] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    impact_sectors: dict[str, float] = Field(default_factory=dict)
    is_geopolitical: bool = False
    regions: list[str] = Field(default_factory=list)
    region: str | None = None
    fingerprint: str = ""
    summary: str | None = None
    model: str


__all__ = [
    "CorrelationOut",
    "GeoEventOut",
    "GeoHeatmapOut",
    "SectorHeatmapItem",
    "SentimentOverviewOut",
    "SentimentSnapshotOut",
    "TextAnalyzeIn",
    "TextAnalyzeOut",
]
