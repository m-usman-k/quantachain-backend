"""API schemas for Module 9. Encrypted secrets never appear here - only ``key_preview``."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.modules.trading.models import (
    ExchangeCredential,
    Fill,
    Order,
    OrderSide,
    OrderSource,
    OrderStatus,
    OrderType,
    PaperAccount,
    Position,
    TradeMode,
    round_quote,
)


# ------------------------------------------------------------------- orders
class OrderCreate(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, pattern=r"^[A-Za-z0-9]+$", description="Base asset, e.g. BTC")
    side: OrderSide
    type: OrderType = OrderType.MARKET
    quantity: float = Field(gt=0, description="Base-asset quantity")
    price: float | None = Field(default=None, gt=0, description="Limit price (limit / stop_limit)")
    stop_price: float | None = Field(default=None, gt=0, description="Trigger price (stop_loss / stop_limit)")
    mode: TradeMode = TradeMode.PAPER
    exchange: str | None = Field(default=None, max_length=32, description="ccxt id; defaults to the server default")
    source: OrderSource = OrderSource.MANUAL
    client_order_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Optional idempotency key; a repeated id returns 409",
    )

    @field_validator("symbol", mode="after")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("exchange", mode="after")
    @classmethod
    def _lower(cls, value: str | None) -> str | None:
        return value.lower().strip() if value else None


class OrderOut(BaseModel):
    id: str
    mode: TradeMode
    exchange: str
    symbol: str
    pair: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: float | None
    stop_price: float | None
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float | None
    fee: float
    fee_asset: str
    client_order_id: str
    exchange_order_id: str | None
    reason: str | None
    source: OrderSource
    created_at: datetime
    updated_at: datetime | None
    triggered_at: datetime | None
    filled_at: datetime | None

    @classmethod
    def from_model(cls, o: Order) -> OrderOut:
        return cls(
            id=o.id or "",
            mode=o.mode,
            exchange=o.exchange,
            symbol=o.symbol,
            pair=o.pair,
            side=o.side,
            type=o.type,
            quantity=o.quantity,
            price=o.price,
            stop_price=o.stop_price,
            status=o.status,
            filled_quantity=o.filled_quantity,
            remaining_quantity=o.remaining_quantity,
            avg_fill_price=o.avg_fill_price,
            fee=o.fee,
            fee_asset=o.fee_asset,
            client_order_id=o.client_order_id,
            exchange_order_id=o.exchange_order_id,
            reason=o.reason,
            source=o.source,
            created_at=o.created_at,
            updated_at=o.updated_at,
            triggered_at=o.triggered_at,
            filled_at=o.filled_at,
        )


class FillOut(BaseModel):
    id: str
    order_id: str
    mode: TradeMode
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    notional: float
    fee: float
    fee_asset: str
    realized_pnl: float
    created_at: datetime

    @classmethod
    def from_model(cls, f: Fill) -> FillOut:
        return cls(
            id=f.id or "",
            order_id=f.order_id,
            mode=f.mode,
            symbol=f.symbol,
            side=f.side,
            quantity=f.quantity,
            price=f.price,
            notional=round_quote(f.quantity * f.price),
            fee=f.fee,
            fee_asset=f.fee_asset,
            realized_pnl=f.realized_pnl,
            created_at=f.created_at,
        )


# ---------------------------------------------------------------- positions
class PositionOut(BaseModel):
    mode: TradeMode
    symbol: str
    quantity: float
    avg_entry_price: float
    cost_basis: float
    realized_pnl: float
    current_price: float | None
    market_value: float | None
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None
    updated_at: datetime | None

    @classmethod
    def from_model(cls, p: Position, price: float | None) -> PositionOut:
        market_value = round_quote(p.quantity * price) if price is not None else None
        unrealized = round_quote((price - p.avg_entry_price) * p.quantity) if price is not None else None
        cost = p.cost_basis
        return cls(
            mode=p.mode,
            symbol=p.symbol,
            quantity=p.quantity,
            avg_entry_price=p.avg_entry_price,
            cost_basis=cost,
            realized_pnl=p.realized_pnl,
            current_price=price,
            market_value=market_value,
            unrealized_pnl=unrealized,
            unrealized_pnl_pct=round(unrealized / cost * 100, 4) if unrealized is not None and cost else None,
            updated_at=p.updated_at or p.created_at,
        )


# ---------------------------------------------------------------- portfolio
class PaperAccountOut(BaseModel):
    quote_asset: str
    cash_balance: float
    starting_balance: float
    realized_pnl: float
    fees_paid: float
    equity_high_water: float
    created_at: datetime
    reset_at: datetime | None

    @classmethod
    def from_model(cls, a: PaperAccount) -> PaperAccountOut:
        return cls(
            quote_asset=a.quote_asset,
            cash_balance=a.cash_balance,
            starting_balance=a.starting_balance,
            realized_pnl=a.realized_pnl,
            fees_paid=a.fees_paid,
            equity_high_water=a.equity_high_water,
            created_at=a.created_at,
            reset_at=a.reset_at,
        )


class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float | None
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float | None
    realized_pnl: float
    allocation_pct: float


class AllocationSlice(BaseModel):
    symbol: str = Field(description='Asset symbol or "CASH"')
    pct: float
    value: float


class PortfolioOut(BaseModel):
    mode: TradeMode
    quote_asset: str
    cash: float
    positions_value: float
    equity: float
    unrealized_pnl: float
    unrealized_pnl_pct: float | None
    realized_pnl: float
    fees_paid: float
    starting_balance: float | None
    total_return_pct: float | None
    day_change_pct: float | None = Field(description="Equity change versus the snapshot taken ~24h ago")
    equity_high_water: float | None
    drawdown_pct: float | None = Field(description="Distance below the equity high-water mark")
    positions: list[PortfolioPosition]
    allocation: list[AllocationSlice]
    note: str | None = None
    updated_at: datetime


class PortfolioPoint(BaseModel):
    ts: datetime
    equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    high: float
    low: float
    samples: int


class PortfolioHistoryOut(BaseModel):
    mode: TradeMode
    range: str
    interval: str
    points: list[PortfolioPoint]
    count: int


class PaperResetOut(BaseModel):
    account: PaperAccountOut
    canceled_orders: int
    closed_positions: int
    portfolio: PortfolioOut


# ------------------------------------------------------------ exchange keys
class ExchangeKeyCreate(BaseModel):
    exchange: str = Field(min_length=2, max_length=32, description="ccxt exchange id, e.g. binance")
    api_key: str = Field(min_length=8, max_length=256)
    secret: str = Field(min_length=8, max_length=512)
    passphrase: str | None = Field(default=None, max_length=256, description="Required by OKX / Coinbase legacy")
    label: str | None = Field(default=None, max_length=60)
    testnet: bool = False

    @field_validator("exchange", mode="after")
    @classmethod
    def _lower(cls, value: str) -> str:
        return value.lower().strip()


class ExchangeKeyOut(BaseModel):
    id: str
    exchange: str
    label: str
    key_preview: str
    testnet: bool
    permissions: list[str]
    is_valid: bool | None
    last_verified_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime | None

    @classmethod
    def from_model(cls, c: ExchangeCredential) -> ExchangeKeyOut:
        return cls(
            id=c.id or "",
            exchange=c.exchange,
            label=c.label,
            key_preview=c.key_preview,
            testnet=c.testnet,
            permissions=c.permissions,
            is_valid=c.is_valid,
            last_verified_at=c.last_verified_at,
            last_error=c.last_error,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )


# ------------------------------------------------------------------- misc
class TradingCapabilities(BaseModel):
    paper: bool = True
    live_enabled: bool
    supported_exchanges: list[str]
    default_exchange: str
    quote_asset: str
    paper_starting_balance: float
    paper_fee_rate: float
    paper_slippage_bps: float
    order_types: list[OrderType]


class TradingSummary(BaseModel):
    open_orders: int
    filled_today: int
    paper_equity: float
    paper_total_return_pct: float | None
    paper_positions: int
    exchange_keys: int
    live_enabled: bool
    generated_at: datetime


class OrderBookOut(BaseModel):
    symbol: str
    pair: str
    bids: list[list[float]] = Field(description="[[price, quantity], ...] best first")
    asks: list[list[float]]
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread: float | None
    spread_pct: float | None
    last_update_id: int | None
    fetched_at: datetime


__all__ = [name for name in globals() if name[0].isupper()]
