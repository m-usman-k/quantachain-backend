"""Module 9 - Trading & Order Execution endpoints (all authenticated; users see only their own data)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import DB, CurrentUser, SettingsDep
from app.core.pagination import Page, Pagination
from app.core.rate_limit import rate_limited
from app.modules.trading.models import OrderStatus, TradeMode
from app.modules.trading.schemas import (
    ExchangeKeyCreate,
    ExchangeKeyOut,
    FillOut,
    OrderBookOut,
    OrderCreate,
    OrderOut,
    PaperResetOut,
    PortfolioHistoryOut,
    PortfolioOut,
    PositionOut,
    TradingCapabilities,
    TradingSummary,
)
from app.modules.trading.service import TradingService

router = APIRouter(prefix="/trading", tags=["trading"])


def get_trading_service(db: DB, settings: SettingsDep) -> TradingService:
    return TradingService(db, settings)


Service = Annotated[TradingService, Depends(get_trading_service)]
order_limiter = rate_limited(60)


@router.get(
    "/mode",
    response_model=TradingCapabilities,
    summary="Trading capabilities (paper always on, live behind a kill switch)",
)
async def mode(_: CurrentUser, service: Service) -> TradingCapabilities:
    return service.capabilities()


@router.get("/summary", response_model=TradingSummary)
async def summary(user: CurrentUser, service: Service) -> TradingSummary:
    return await service.summary(user)


# -------------------------------------------------------------------- orders
@router.post(
    "/orders",
    response_model=OrderOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(order_limiter)],
    summary="Place a market, limit, stop-loss or stop-limit order",
)
async def create_order(payload: OrderCreate, user: CurrentUser, service: Service) -> OrderOut:
    return OrderOut.from_model(await service.create_order(user, payload))


@router.get("/orders", response_model=Page[OrderOut], summary="My orders, newest first")
async def list_orders(
    user: CurrentUser,
    service: Service,
    page: Pagination,
    mode: Annotated[TradeMode | None, Query()] = None,
    status_: Annotated[OrderStatus | None, Query(alias="status")] = None,
    symbol: Annotated[str | None, Query()] = None,
) -> Page[OrderOut]:
    items, total = await service.list_orders(user, page, mode=mode, status=status_, symbol=symbol)
    return Page.build([OrderOut.from_model(o) for o in items], total, page)


@router.get("/orders/{order_id}", response_model=OrderOut)
async def get_order(order_id: str, user: CurrentUser, service: Service) -> OrderOut:
    return OrderOut.from_model(await service.get_order(user, order_id))


@router.delete("/orders/{order_id}", response_model=OrderOut, summary="Cancel a working order")
async def cancel_order(order_id: str, user: CurrentUser, service: Service) -> OrderOut:
    return OrderOut.from_model(await service.cancel_order(user, order_id))


@router.get("/fills", response_model=Page[FillOut])
async def list_fills(
    user: CurrentUser,
    service: Service,
    page: Pagination,
    mode: Annotated[TradeMode | None, Query()] = None,
    symbol: Annotated[str | None, Query()] = None,
) -> Page[FillOut]:
    items, total = await service.list_fills(user, page, mode=mode, symbol=symbol)
    return Page.build([FillOut.from_model(f) for f in items], total, page)


# ----------------------------------------------------------------- portfolio
@router.get("/positions", response_model=list[PositionOut])
async def positions(
    user: CurrentUser, service: Service, mode: Annotated[TradeMode, Query()] = TradeMode.PAPER
) -> list[PositionOut]:
    return await service.positions_out(user, mode)


@router.get("/portfolio", response_model=PortfolioOut, summary="Equity, PnL and allocation")
async def portfolio(
    user: CurrentUser, service: Service, mode: Annotated[TradeMode, Query()] = TradeMode.PAPER
) -> PortfolioOut:
    return await service.portfolio_out(user, mode)


@router.get("/portfolio/history", response_model=PortfolioHistoryOut, summary="Equity curve")
async def portfolio_history(
    user: CurrentUser,
    service: Service,
    mode: Annotated[TradeMode, Query()] = TradeMode.PAPER,
    range: Annotated[str, Query(pattern=r"^\d+[mhdw]$")] = "7d",
    interval: Annotated[str, Query(pattern="^(1m|5m|15m|1h|4h|1d)$")] = "1h",
) -> PortfolioHistoryOut:
    return await service.portfolio_history(user, mode, range_=range, interval=interval)


@router.post("/paper/reset", response_model=PaperResetOut, summary="Reset the paper account to its starting balance")
async def paper_reset(user: CurrentUser, service: Service) -> PaperResetOut:
    return await service.paper_reset(user)


@router.get("/orderbook/{symbol}", response_model=OrderBookOut, summary="Live order book (public market data)")
async def orderbook(
    symbol: str, _: CurrentUser, service: Service, limit: Annotated[int, Query(ge=5, le=100)] = 20
) -> OrderBookOut:
    return await service.orderbook(symbol, limit=limit)


# ------------------------------------------------------------- exchange keys
@router.get("/exchange-keys", response_model=list[ExchangeKeyOut])
async def list_keys(user: CurrentUser, service: Service) -> list[ExchangeKeyOut]:
    return [ExchangeKeyOut.from_model(c) for c in await service.list_keys(user)]


@router.post(
    "/exchange-keys",
    response_model=ExchangeKeyOut,
    status_code=status.HTTP_201_CREATED,
    summary="Store exchange API keys (encrypted at rest)",
)
async def add_key(payload: ExchangeKeyCreate, user: CurrentUser, service: Service) -> ExchangeKeyOut:
    return ExchangeKeyOut.from_model(await service.add_key(user, payload))


@router.post("/exchange-keys/{credential_id}/verify", response_model=ExchangeKeyOut)
async def verify_key(credential_id: str, user: CurrentUser, service: Service) -> ExchangeKeyOut:
    return ExchangeKeyOut.from_model(await service.verify_key(user, credential_id))


@router.delete("/exchange-keys/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_key(credential_id: str, user: CurrentUser, service: Service) -> None:
    await service.delete_key(user, credential_id)
