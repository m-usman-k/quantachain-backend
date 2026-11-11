"""Paper trading engine (FE-1, always on).

Matching model
--------------
* **Market** orders fill immediately at the last price moved ``SLIPPAGE_BPS`` (5 bps)
  against the trader: buys pay ``last * 1.0005``, sells receive ``last * 0.9995``.
* **Limit** buys fill when ``price <= limit`` and limit sells when ``price >= limit``,
  always at the limit price (conservative: no price improvement is simulated).
* **Stop-loss** orders arm at placement; a sell triggers when ``price <= stop`` and a
  buy when ``price >= stop``, then fills as a market order at the triggering price.
* **Stop-limit** orders trigger the same way and then rest as a limit order.
* Every fill pays ``settings.paper_fee_rate`` of the notional in the quote asset.
* No funds are reserved for resting orders: cash / position checks run when the order
  is placed (synchronous rejection) and again, atomically, when it fills.

Idempotency: the stream matcher and the sweep job may evaluate the same order at the
same time. A fill first *claims* the order with an atomic status transition
(``working -> filled``); the loser of that race sees ``None`` and stops. Cash and
positions are then settled with compare-and-set writes, so money is never spent twice.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.events import EventBus, Topics, event_bus
from app.core.exceptions import ConflictError
from app.core.timeutils import utcnow
from app.modules.market.models import SourceStatus
from app.modules.market.service import MarketService
from app.modules.trading.models import (
    WORKING_STATUSES,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PaperAccount,
    TradeMode,
    round_price,
    round_quote,
)
from app.modules.trading.portfolio import Ledger, PortfolioService
from app.modules.trading.repository import (
    OrderRepository,
    PaperAccountRepository,
    PositionRepository,
    StaleWriteError,
)
from app.workers.sources import SourceTracker

logger = structlog.get_logger(__name__)

SLIPPAGE_BPS = 5.0
SOURCE_NAME = "paper_engine"
SYMBOL_CACHE_SECONDS = 30.0


def slipped_price(last_price: float, side: OrderSide) -> float:
    """Market-fill price: ``SLIPPAGE_BPS`` against the trader."""
    factor = 1 + SLIPPAGE_BPS / 10_000 if side == OrderSide.BUY else 1 - SLIPPAGE_BPS / 10_000
    return round_price(last_price * factor)


def limit_crosses(order: Order, price: float) -> bool:
    if order.price is None:
        return False
    return price <= order.price if order.side == OrderSide.BUY else price >= order.price


def stop_crosses(order: Order, price: float) -> bool:
    if order.stop_price is None:
        return False
    return price <= order.stop_price if order.side == OrderSide.SELL else price >= order.stop_price


class PaperEngine:
    def __init__(
        self,
        db: AsyncDatabase[dict[str, Any]],
        settings: Settings | None = None,
        *,
        bus: EventBus | None = None,
        ledger: Ledger | None = None,
        portfolio: PortfolioService | None = None,
        tracker: SourceTracker | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.orders = OrderRepository(db)
        self.positions = PositionRepository(db)
        self.accounts = PaperAccountRepository(db)
        self.ledger = ledger or Ledger(db, self.settings, self.bus)
        self.portfolio = portfolio or PortfolioService(db, self.settings, self.bus)
        self.tracker = tracker

    @property
    def fee_rate(self) -> float:
        return self.settings.paper_fee_rate

    # ------------------------------------------------------------- helpers
    async def account(self, user_id: str) -> PaperAccount:
        return await self.accounts.get_or_create(
            user_id, quote_asset=self.settings.quote_asset, starting_balance=self.settings.paper_starting_balance
        )

    def fee_for(self, notional: float) -> float:
        return round_quote(notional * self.fee_rate)

    async def check_funds(self, order: Order, quantity: float, price: float) -> str | None:
        """Return a rejection reason when the account cannot back this fill, else ``None``."""
        if order.side == OrderSide.BUY:
            notional = round_quote(quantity * price)
            need = round_quote(notional + self.fee_for(notional))
            account = await self.account(order.user_id)
            if account.cash_balance + 1e-9 < need:
                return (
                    f"Insufficient cash: need {need:,.2f} {account.quote_asset}, "
                    f"available {account.cash_balance:,.2f} {account.quote_asset}"
                )
            return None
        position = await self.positions.get_position(order.user_id, order.mode, order.symbol)
        held = position.quantity if position else 0.0
        if held + 1e-12 < quantity:
            return f"Insufficient position: holding {held:g} {order.symbol}, selling {quantity:g}"
        return None

    def _track_fill(self) -> None:
        if self.tracker is not None and SOURCE_NAME in self.tracker.sources:
            self.tracker.event(SOURCE_NAME)

    # ----------------------------------------------------------- lifecycle
    async def place(self, order: Order, last_price: float | None) -> Order:
        """Evaluate a freshly inserted (``new``) paper order. Returns its state after matching."""
        await self.account(order.user_id)
        if order.type == OrderType.MARKET:
            if last_price is None:
                return await self._reject(order, f"No market price available for {order.symbol}")
            filled = await self._fill(order, price=slipped_price(last_price, order.side))
            return filled or await self._current(order)

        reference = order.price if order.type == OrderType.LIMIT else order.stop_price
        reason = await self.check_funds(order, order.quantity, reference or 0.0)
        if reason:
            return await self._reject(order, reason)
        if last_price is None:
            return await self._open(order) or await self._current(order)
        return await self.evaluate(order, last_price) or await self._current(order)

    async def evaluate(self, order: Order, price: float) -> Order | None:
        """Advance one working order against ``price``. Returns the order when its state changed."""
        if order.status not in WORKING_STATUSES or order.mode != TradeMode.PAPER:
            return None
        if order.type == OrderType.MARKET:
            return await self._fill(order, price=slipped_price(price, order.side))
        if order.type == OrderType.LIMIT:
            if limit_crosses(order, price):
                return await self._fill(order, price=order.price or price)
            return await self._open(order)
        if order.type == OrderType.STOP_LOSS:
            if stop_crosses(order, price):
                return await self._fill(order, price=slipped_price(price, order.side), triggered=True)
            return await self._open(order)

        # stop_limit: arm -> trigger -> rest as a limit order
        changed: Order | None = None
        if order.status in (OrderStatus.NEW, OrderStatus.OPEN):
            if not stop_crosses(order, price):
                return await self._open(order)
            triggered = await self.orders.transition(
                order.id or "",
                from_statuses={OrderStatus.NEW, OrderStatus.OPEN},
                to_status=OrderStatus.TRIGGERED,
                fields={"triggered_at": utcnow()},
            )
            if triggered is None:
                return None
            self.ledger.publish_order(triggered)
            order = changed = triggered
        if limit_crosses(order, price):
            return await self._fill(order, price=order.price or price) or changed
        return changed

    async def evaluate_open_orders(self, symbol: str, price: float) -> int:
        """Re-check every working paper order of ``symbol``. Returns the number of fills."""
        fills = 0
        for order in await self.orders.working(mode=TradeMode.PAPER, symbol=symbol):
            try:
                updated = await self.evaluate(order, price)
            except StaleWriteError as exc:
                logger.warning("paper_evaluate_contended", order_id=order.id, error=str(exc))
                continue
            if updated is not None and updated.status == OrderStatus.FILLED:
                fills += 1
        return fills

    async def cancel(self, order: Order, *, reason: str = "Canceled by user") -> Order:
        canceled = await self.orders.transition(
            order.id or "", from_statuses=WORKING_STATUSES, to_status=OrderStatus.CANCELED, fields={"reason": reason}
        )
        if canceled is None:
            current = await self._current(order)
            raise ConflictError(f"Order is already {current.status}")
        self.ledger.publish_order(canceled)
        return canceled

    async def reset(self, user_id: str) -> tuple[PaperAccount, int, int]:
        """Cancel working orders, drop positions and restore the starting balance."""
        canceled = 0
        for order in await self.orders.working(mode=TradeMode.PAPER, user_id=user_id):
            done = await self.orders.transition(
                order.id or "",
                from_statuses=WORKING_STATUSES,
                to_status=OrderStatus.CANCELED,
                fields={"reason": "Paper account reset"},
            )
            if done is not None:
                canceled += 1
                self.ledger.publish_order(done)
        closed = await self.positions.delete_many({"user_id": user_id, "mode": str(TradeMode.PAPER)})
        account = await self.accounts.reset(
            user_id, quote_asset=self.settings.quote_asset, starting_balance=self.settings.paper_starting_balance
        )
        return account, canceled, closed

    # ------------------------------------------------------------ internals
    async def _current(self, order: Order) -> Order:
        return await self.orders.get(order.id or "") or order

    async def _open(self, order: Order) -> Order | None:
        if order.status != OrderStatus.NEW:
            return None
        opened = await self.orders.transition(
            order.id or "", from_statuses={OrderStatus.NEW}, to_status=OrderStatus.OPEN
        )
        if opened is not None:
            self.ledger.publish_order(opened)
        return opened

    async def _reject(self, order: Order, reason: str, *, notify: bool = False) -> Order:
        rejected = await self.orders.transition(
            order.id or "", from_statuses=WORKING_STATUSES, to_status=OrderStatus.REJECTED, fields={"reason": reason}
        )
        if rejected is None:
            return await self._current(order)
        self.ledger.publish_order(rejected)
        if notify:
            await self.ledger.notify_rejected(rejected)
        return rejected

    async def _unfill(self, order: Order, reason: str) -> Order:
        """Undo a claimed fill whose settlement failed (lost a cash / position race)."""
        reverted = await self.orders.transition(
            order.id or "",
            from_statuses={OrderStatus.FILLED},
            to_status=OrderStatus.REJECTED,
            fields={"reason": reason, "filled_quantity": 0.0, "avg_fill_price": None, "fee": 0.0, "filled_at": None},
        )
        reverted = reverted or await self._current(order)
        self.ledger.publish_order(reverted)
        await self.ledger.notify_rejected(reverted)
        return reverted

    async def _fill(self, order: Order, *, price: float, triggered: bool = False) -> Order | None:
        quantity = order.remaining_quantity
        notional = round_quote(quantity * price)
        fee = self.fee_for(notional)
        was_resting = order.status != OrderStatus.NEW

        reason = await self.check_funds(order, quantity, price)
        if reason:
            return await self._reject(order, reason, notify=was_resting)

        now = utcnow()
        fields: dict[str, Any] = {
            "filled_quantity": quantity,
            "avg_fill_price": price,
            "fee": fee,
            "filled_at": now,
            "reason": None,
        }
        if triggered:
            fields["triggered_at"] = now
        claimed = await self.orders.transition(
            order.id or "", from_statuses=WORKING_STATUSES, to_status=OrderStatus.FILLED, fields=fields
        )
        if claimed is None:
            return None  # another evaluator filled or canceled it first

        if order.side == OrderSide.BUY:
            account = await self.accounts.adjust(order.user_id, cash_delta=-(notional + fee), fee_delta=fee)
            if account is None:
                return await self._unfill(claimed, "Insufficient cash at fill time")
            _, realized = await self.ledger.apply_position(claimed, quantity, price, fee)
        else:
            position, realized = await self.ledger.apply_position(claimed, quantity, price, fee)
            if position is None:
                return await self._unfill(claimed, "Insufficient position at fill time")
            await self.accounts.adjust(
                order.user_id,
                cash_delta=notional - fee,
                realized_delta=realized,
                fee_delta=fee,
                require_cash=False,
            )

        fill = await self.ledger.insert_fill(claimed, quantity, price, fee, realized)
        self.ledger.publish_order(claimed)
        await self.portfolio.publish(order.user_id, TradeMode.PAPER)
        await self.ledger.notify_fill(claimed, fill)
        self._track_fill()
        logger.info(
            "paper_order_filled",
            order_id=claimed.id,
            user_id=claimed.user_id,
            symbol=claimed.symbol,
            side=str(claimed.side),
            quantity=quantity,
            price=price,
            fee=fee,
        )
        return claimed


class PaperMatcher:
    """Worker-side driver: re-checks paper orders on every price tick and in a periodic sweep.

    A 30-second cache of symbols that currently have working orders keeps the tick
    path free of database queries for symbols nobody is trading.
    """

    def __init__(
        self,
        db: AsyncDatabase[dict[str, Any]],
        settings: Settings | None = None,
        *,
        tracker: SourceTracker | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.tracker = tracker
        self.engine = PaperEngine(db, self.settings, bus=self.bus, tracker=tracker)
        self.market = MarketService(db, self.settings)
        self._symbols: set[str] = set()
        self._symbols_at = 0.0

    async def symbols(self, *, force: bool = False) -> set[str]:
        if force or time.monotonic() - self._symbols_at > SYMBOL_CACHE_SECONDS:
            self._symbols = await self.engine.orders.working_symbols(TradeMode.PAPER)
            self._symbols_at = time.monotonic()
        return self._symbols

    def invalidate(self) -> None:
        self._symbols_at = 0.0

    def _enabled(self) -> bool:
        return self.tracker is None or self.tracker.enabled(SOURCE_NAME)

    async def handle_tick(self, payload: dict[str, Any]) -> int:
        symbol = str(payload.get("symbol") or "").upper()
        price = payload.get("price")
        if not symbol or price is None:
            return 0
        if symbol not in await self.symbols():
            return 0
        fills = await self.engine.evaluate_open_orders(symbol, float(price))
        if fills:
            self.invalidate()
        return fills

    async def run(self) -> None:
        """Stream factory for ``runtime.add_stream``: consume ``Topics.PRICE_TICK`` forever."""
        subscription = self.bus.subscribe(Topics.PRICE_TICK, maxsize=5000)
        if self.tracker is not None:
            self.tracker.set_status(SOURCE_NAME, SourceStatus.ACTIVE, message="Paper matcher listening to price ticks")
        try:
            while True:
                event = await subscription.get(timeout=5.0)
                if event is None or not self._enabled():
                    continue
                try:
                    await self.handle_tick(event.payload)
                except Exception as exc:  # keep consuming ticks through transient DB errors
                    logger.warning("paper_matcher_tick_failed", error=str(exc))
                    if self.tracker is not None:
                        self.tracker.error(SOURCE_NAME, f"{type(exc).__name__}: {exc}")
        finally:
            subscription.close()
            if self.tracker is not None and self.tracker.sources[SOURCE_NAME].status == SourceStatus.ACTIVE:
                self.tracker.set_status(SOURCE_NAME, SourceStatus.STOPPED)

    async def sweep(self) -> dict[str, Any]:
        """Safety net job: evaluate working paper orders against the latest stored price."""
        if not self._enabled():
            return {"skipped": "disabled"}
        symbols = await self.symbols(force=True)
        evaluated = 0
        fills = 0
        for symbol in sorted(symbols):
            price = await self.market.latest_price(symbol)
            if price is None:
                continue
            evaluated += 1
            fills += await self.engine.evaluate_open_orders(symbol, price)
        if fills:
            self.invalidate()
        return {"symbols": len(symbols), "evaluated": evaluated, "fills": fills}


__all__ = [
    "SLIPPAGE_BPS",
    "SOURCE_NAME",
    "SYMBOL_CACHE_SECONDS",
    "PaperEngine",
    "PaperMatcher",
    "limit_crosses",
    "slipped_price",
    "stop_crosses",
]
