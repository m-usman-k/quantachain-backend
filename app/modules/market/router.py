"""Module 2 - Multi-Source Data Aggregation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import DB, AdminUser, SettingsDep
from app.core.pagination import Page, Pagination
from app.modules.market.models import Asset
from app.modules.market.schemas import (
    AddAssetRequest,
    AssetDetail,
    AssetRow,
    CandleSeries,
    IngestionOverview,
    IngestionSourceOut,
    LogLine,
    MarketOverview,
    NewsOut,
    SocialOut,
    Ticker,
    ToggleRequest,
)
from app.modules.market.service import MarketService

router = APIRouter(prefix="/market", tags=["market"])


def get_market_service(db: DB, settings: SettingsDep) -> MarketService:
    return MarketService(db, settings)


Service = Annotated[MarketService, Depends(get_market_service)]


@router.get("/overview", response_model=MarketOverview, summary="Dashboard header: market pulse + top assets")
async def overview(service: Service, limit: Annotated[int, Query(ge=1, le=50)] = 10) -> MarketOverview:
    return await service.overview(limit=limit)


@router.get("/assets", response_model=list[AssetRow], summary="Markets table (Hot / Gainers / Losers / Volume / New)")
async def list_assets(
    service: Service,
    tab: Annotated[str, Query(pattern="^(hot|gainers|losers|volume|new|market_cap)$")] = "hot",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AssetRow]:
    return await service.list_assets(tab, limit=limit)


@router.post("/assets", response_model=Asset, status_code=status.HTTP_201_CREATED, summary="Track a new asset (admin)")
async def add_asset(payload: AddAssetRequest, _: AdminUser, service: Service) -> Asset:
    return await service.add_asset(
        payload.symbol, name=payload.name, coingecko_id=payload.coingecko_id, contracts=payload.contracts
    )


@router.delete("/assets/{symbol}", status_code=status.HTTP_204_NO_CONTENT, summary="Stop tracking an asset (admin)")
async def remove_asset(symbol: str, _: AdminUser, service: Service) -> None:
    await service.remove_asset(symbol)


@router.get("/assets/{symbol}", response_model=AssetDetail, summary="Asset profile with latest stats")
async def asset_detail(symbol: str, service: Service) -> AssetDetail:
    return await service.asset_detail(symbol)


@router.get("/assets/{symbol}/ticker", response_model=Ticker, summary="Latest price and 24h statistics")
async def ticker(symbol: str, service: Service) -> Ticker:
    return await service.ticker(symbol)


@router.get("/assets/{symbol}/candles", response_model=CandleSeries, summary="OHLCV candles")
async def candles(
    symbol: str,
    service: Service,
    interval: Annotated[str, Query(pattern="^(1m|5m|15m|1h|4h|1d)$")] = "1h",
    limit: Annotated[int, Query(ge=1, le=1500)] = 500,
    start: Annotated[datetime | None, Query(description="Inclusive lower bound on open time")] = None,
    end: Annotated[datetime | None, Query(description="Inclusive upper bound on open time")] = None,
) -> CandleSeries:
    rows = await service.candle_series(symbol, interval, limit=limit, start=start, end=end)
    return CandleSeries(symbol=symbol.upper(), interval=interval, candles=rows, count=len(rows))


@router.get("/news", response_model=Page[NewsOut], summary="Aggregated news with sentiment analysis")
async def news(
    service: Service,
    page: Pagination,
    symbol: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query(description="Full-text search")] = None,
    source: Annotated[str | None, Query()] = None,
    analyzed_only: Annotated[bool, Query()] = False,
    geopolitical_only: Annotated[bool, Query()] = False,
) -> Page[NewsOut]:
    items, total = await service.list_news(
        page, symbol=symbol, query=q, source=source, analyzed_only=analyzed_only, geopolitical_only=geopolitical_only
    )
    return Page.build([NewsOut.from_model(a) for a in items], total, page)


@router.get("/news/{article_id}", response_model=NewsOut)
async def article(article_id: str, service: Service) -> NewsOut:
    return NewsOut.from_model(await service.get_article(article_id))


@router.get("/social", response_model=Page[SocialOut], summary="Deduplicated social signals")
async def social(
    service: Service,
    page: Pagination,
    symbol: Annotated[str | None, Query()] = None,
    platform: Annotated[str | None, Query()] = None,
    community: Annotated[str | None, Query()] = None,
) -> Page[SocialOut]:
    items, total = await service.list_social(page, symbol=symbol, platform=platform, community=community)
    return Page.build([SocialOut.from_model(p) for p in items], total, page)


# ------------------------------------------------------------ ingestion hub
@router.get("/ingestion", response_model=IngestionOverview, summary="Data Ingestion Hub status (admin)")
async def ingestion(_: AdminUser, service: Service) -> IngestionOverview:
    return await service.ingestion_overview()


@router.post(
    "/ingestion/sources/{name}", response_model=IngestionSourceOut, summary="Enable or disable a source (admin)"
)
async def toggle_source(name: str, payload: ToggleRequest, _: AdminUser, service: Service) -> IngestionSourceOut:
    return await service.set_source_enabled(name, payload.enabled)


@router.get("/ingestion/logs", response_model=list[LogLine], summary="Recent live ingestion log lines (admin)")
async def ingestion_logs(
    _: AdminUser, service: Service, limit: Annotated[int, Query(ge=1, le=500)] = 100
) -> list[LogLine]:
    return await service.recent_logs(limit)
