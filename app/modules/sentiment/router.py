"""Module 4 - Sentiment & Geopolitical Analysis endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import DB, SettingsDep
from app.core.pagination import Page, Pagination
from app.modules.sentiment.models import GeoCategory
from app.modules.sentiment.schemas import (
    CorrelationOut,
    GeoEventOut,
    GeoHeatmapOut,
    SentimentOverviewOut,
    SentimentSnapshotOut,
    TextAnalyzeIn,
    TextAnalyzeOut,
)
from app.modules.sentiment.service import SentimentService

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


def get_sentiment_service(db: DB, settings: SettingsDep) -> SentimentService:
    return SentimentService(db, settings)


Service = Annotated[SentimentService, Depends(get_sentiment_service)]


@router.get("/overview", response_model=SentimentOverviewOut, summary="Market sentiment overview & Vibe Index")
async def overview(service: Service) -> SentimentOverviewOut:
    return await service.get_overview()


@router.get(
    "/history", response_model=list[SentimentSnapshotOut], summary="Historical sentiment snapshots for chart series"
)
async def history(
    service: Service,
    symbol: Annotated[str | None, Query()] = None,
    days: Annotated[int, Query(ge=1, le=90)] = 7,
) -> list[SentimentSnapshotOut]:
    return await service.get_history(symbol=symbol.upper() if symbol else None, days=days)


@router.get(
    "/heatmap", response_model=GeoHeatmapOut, summary="Geopolitical impact heatmap across crypto sectors and regions"
)
async def heatmap(
    service: Service,
    days: Annotated[int, Query(ge=1, le=30)] = 7,
) -> GeoHeatmapOut:
    return await service.get_heatmap(days=days)


@router.get("/events", response_model=Page[GeoEventOut], summary="List geopolitical and regulatory news events")
async def events(
    service: Service,
    pagination: Pagination,
    category: Annotated[GeoCategory | None, Query()] = None,
    sector: Annotated[str | None, Query()] = None,
    region: Annotated[str | None, Query()] = None,
) -> Page[GeoEventOut]:
    items, total = await service.geo_events.find_events(
        category=category,
        sector=sector,
        region=region,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return Page.build([GeoEventOut(**e.model_dump()) for e in items], total, pagination)


@router.get(
    "/correlation",
    response_model=CorrelationOut,
    summary="Programmatically correlate sentiment with historical price movements",
)
async def correlation(
    service: Service,
    symbol: Annotated[str, Query()] = "BTC",
    days: Annotated[int, Query(ge=7, le=180)] = 30,
) -> CorrelationOut:
    return await service.compute_correlation(symbol=symbol, days=days)


@router.post(
    "/analyze",
    response_model=TextAnalyzeOut,
    summary="Analyze arbitrary crypto text for sentiment, NER entities, and sector impact",
)
async def analyze_text(payload: TextAnalyzeIn, service: Service) -> TextAnalyzeOut:
    analysis = await service.analyze_text(payload.text, symbol_hint=payload.symbol)
    return TextAnalyzeOut(
        sentiment_score=analysis.sentiment_score or 0.0,
        sentiment_label=analysis.sentiment_label or "neutral",  # type: ignore[arg-type]
        confidence=analysis.confidence or 0.5,
        entities=analysis.entities,
        topics=analysis.topics,
        impact_sectors=analysis.impact_sectors,
        is_geopolitical=analysis.is_geopolitical,
        regions=analysis.regions,
        summary=analysis.summary,
        model=analysis.model or "heuristic",
    )
