"""Module 9 persistence models: orders, fills, positions, paper accounts, exchange credentials.

Money math convention (shared by every file in this module)
-----------------------------------------------------------
Documents store plain floats. To keep them free of binary noise every value is
rounded with an explicit helper before it is written:

* quantities (base asset)                 -> ``round_qty``   (8 dp)
* prices                                  -> ``round_price`` (8 dp, sub-dollar assets need them)
* quote-currency money (notional, fees,
  balances, realised / unrealised PnL)    -> ``round_quote`` (2 dp)

Balances are never ``$inc``-ed; they are updated with compare-and-set writes of
already-rounded values (see ``repository.py``), so what is stored is exactly what
the API returns.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import Field

from app.db.models import Document

QTY_DECIMALS = 8
PRICE_DECIMALS = 8
QUOTE_DECIMALS = 2

# Anything below this is treated as a zero quantity (float dust after subtraction).
QTY_EPSILON = 10**-QTY_DECIMALS


def round_qty(value: float) -> float:
    return round(float(value), QTY_DECIMALS)


def round_price(value: float) -> float:
    return round(float(value), PRICE_DECIMALS)


def round_quote(value: float) -> float:
    return round(float(value), QUOTE_DECIMALS)


class TradeMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderStatus(StrEnum):
    NEW = "new"  # accepted, not yet evaluated against a price
    OPEN = "open"  # resting (limit) or armed (stop)
    TRIGGERED = "triggered"  # stop reached; stop_limit now rests as a limit order
    PARTIALLY_FILLED = "partially_filled"  # live orders only
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"  # the engine / exchange refused it (see ``reason``)
    FAILED = "failed"  # transport or exchange error while submitting a live order


class OrderSource(StrEnum):
    MANUAL = "manual"
    SIGNAL = "signal"
    API = "api"


FINAL_STATUSES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.FAILED}
)
# Orders that can still change state: evaluated by the matcher and cancelable by the user.
WORKING_STATUSES: frozenset[OrderStatus] = frozenset(
    {OrderStatus.NEW, OrderStatus.OPEN, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED}
)
STOP_TYPES: frozenset[OrderType] = frozenset({OrderType.STOP_LOSS, OrderType.STOP_LIMIT})
LIMIT_TYPES: frozenset[OrderType] = frozenset({OrderType.LIMIT, OrderType.STOP_LIMIT})


def new_client_order_id() -> str:
    return uuid.uuid4().hex


class Order(Document):
    user_id: str
    mode: TradeMode = TradeMode.PAPER
    exchange: str  # ccxt exchange id, e.g. "binance"
    symbol: str  # base asset, e.g. BTC
    pair: str  # unified pair, e.g. BTC/USDT
    side: OrderSide
    type: OrderType
    quantity: float
    price: float | None = None  # limit price (limit / stop_limit)
    stop_price: float | None = None  # trigger price (stop_loss / stop_limit)
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    fee: float = 0.0
    fee_asset: str = "USDT"
    client_order_id: str = Field(default_factory=new_client_order_id)
    exchange_order_id: str | None = None
    reason: str | None = None
    source: OrderSource = OrderSource.MANUAL
    triggered_at: datetime | None = None
    filled_at: datetime | None = None

    @property
    def is_final(self) -> bool:
        return self.status in FINAL_STATUSES

    @property
    def remaining_quantity(self) -> float:
        return round_qty(max(self.quantity - self.filled_quantity, 0.0))


class Fill(Document):
    order_id: str
    user_id: str
    mode: TradeMode
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_asset: str = "USDT"
    realized_pnl: float = 0.0  # sells only: (price - avg_entry) * quantity - fee
    exchange_trade_id: str | None = None


class Position(Document):
    """Net long holding of one asset in one mode. Weighted-average entry accounting."""

    user_id: str
    mode: TradeMode
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0  # lifetime realised PnL for this symbol (survives flat periods)

    @property
    def is_open(self) -> bool:
        return self.quantity > QTY_EPSILON

    @property
    def cost_basis(self) -> float:
        return round_quote(self.quantity * self.avg_entry_price)


class PaperAccount(Document):
    """Simulated quote-currency wallet. One per user, created on first use."""

    user_id: str
    quote_asset: str = "USDT"
    cash_balance: float
    starting_balance: float
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    reset_at: datetime | None = None
    equity_high_water: float = 0.0


class ExchangeCredential(Document):
    """User API keys for a real exchange. Secrets are Fernet-encrypted at rest (``SecretBox``)."""

    user_id: str
    exchange: str  # ccxt id
    label: str = ""
    api_key_encrypted: str
    secret_encrypted: str
    passphrase_encrypted: str | None = None
    key_preview: str  # "abcd…wxyz" - the only part of the key ever returned by the API
    testnet: bool = False
    permissions: list[str] = Field(default_factory=list)
    is_valid: bool | None = None  # None = never verified (live trading disabled or provider unreachable)
    last_verified_at: datetime | None = None
    last_error: str | None = None


__all__ = [
    "FINAL_STATUSES",
    "LIMIT_TYPES",
    "PRICE_DECIMALS",
    "QTY_DECIMALS",
    "QTY_EPSILON",
    "QUOTE_DECIMALS",
    "STOP_TYPES",
    "WORKING_STATUSES",
    "ExchangeCredential",
    "Fill",
    "Order",
    "OrderSide",
    "OrderSource",
    "OrderStatus",
    "OrderType",
    "PaperAccount",
    "Position",
    "TradeMode",
    "new_client_order_id",
    "round_price",
    "round_qty",
    "round_quote",
]
