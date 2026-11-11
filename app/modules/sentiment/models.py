"""Module 4 persistence models: geopolitical events, sentiment snapshots, and correlation analysis."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.core.timeutils import utcnow
from app.db.models import Document
from app.modules.market.models import SentimentLabel


class GeoCategory(StrEnum):
    REGULATION = "regulation"
    SECURITY = "security"
    MACRO = "macro"
    LEGAL = "legal"
    GEOPOLITICAL = "geopolitical"
    GENERAL = "general"


class GeoEvent(Document):
    """Geopolitical or regulatory news event impacting crypto markets."""

    title: str
    category: GeoCategory = GeoCategory.GENERAL
    impacted_sectors: list[str] = Field(default_factory=list)  # DeFi, Layer 1, Stablecoins, CeFi, NFT
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    summary: str = ""
    regions: list[str] = Field(default_factory=list)  # US, EU, CN, AE, etc.
    region: str | None = None
    fingerprint: str = ""
    source_url: str | None = None
    source_name: str | None = None
    published_at: datetime
    occurred_at: datetime = Field(default_factory=utcnow)
    observed_associations: list[dict[str, Any]] = Field(default_factory=list)


class SentimentSnapshot(Document):
    """Cross-module time-series contract: see docs/ARCHITECTURE.md."""

    ts: datetime
    meta: dict[str, Any]  # {"scope": "global" | "asset", "symbol": "BTC" | None}
    score: float = Field(ge=-1.0, le=1.0)
    index: int = Field(ge=0, le=100)  # Vibe Index (0 = Extreme Fear, 100 = Extreme Greed)
    label: SentimentLabel
    sample_size: int = 0
    breakdown: dict[str, float] = Field(default_factory=dict)  # retail, institutional, whale
    confidence: float = 1.0


class CorrelationResult(BaseModel):
    symbol: str
    window_days: int
    correlation: float
    sample_count: int
    interpretation: str
    computed_at: datetime


__all__ = [
    "CorrelationResult",
    "GeoCategory",
    "GeoEvent",
    "SentimentLabel",
    "SentimentSnapshot",
]
