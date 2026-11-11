"""Portfolio accounting for Module 9 (FE-2).

* ``Ledger``           - bookkeeping shared by the paper engine and the live executor:
                         position updates, ``fills`` rows, ``Topics.ORDER`` events and
                         order-update notifications.
* ``PortfolioService`` - valuation (equity, unrealised / realised PnL, allocation),
                         time-series snapshots (``portfolio_snapshots``) and bucketed history.

Valuation rules: ``equity = cash + positions_value``; a position without a known price
is valued at cost (so it contributes zero unrealised PnL rather than vanishing).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.events import EventBus, Topics, event_bus
from app.core.exceptions import ValidationFailedError
from app.core.timeutils import parse_range, utcnow
from app.modules.auth.repository import UserRepository
from app.modules.market.repository import AssetStatsRepository
from app.modules.market.service import MarketService
from app.modules.notifications.models import NotificationType, Severity
from app.modules.notifications.service import NotificationService
from app.modules.trading.models import (
    Fill,
    Order,
    OrderSide,
    Position,
    TradeMode,
    round_price,
    round_qty,
    round_quote,
)
from app.modules.trading.repository import (
    FillRepository,
    OrderRepository,
    PaperAccountRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
)
from app.modules.trading.schemas import (
    AllocationSlice,
    PortfolioHistoryOut,
    PortfolioOut,
    PortfolioPoint,
    PortfolioPosition,
)

logger = structlog.get_logger(__name__)

DAY = timedelta(hours=24)
ACTIVE_WINDOW = timedelta(days=7)
MAX_HISTORY = timedelta(days=366)


def order_event(order: Order) -> dict[str, Any]:
    """``Topics.ORDER`` payload (JSON-serialisable; ``user_id`` lets ``/ws/portfolio`` filter)."""
    return {
        "user_id": order.user_id,
        "order_id": order.id,
        "client_order_id": order.client_order_id,
        "mode": str(order.mode),
        "symbol": order.symbol,
        "pair": order.pair,
        "status": str(order.status),
        "side": str(order.side),
        "type": str(order.type),
        "quantity": order.quantity,
        "filled_quantity": order.filled_quantity,
        "avg_fill_price": order.avg_fill_price,
        "fee": order.fee,
        "reason": order.reason,
        "updated_at": (order.updated_at or order.created_at).isoformat(),
    }


class Ledger:
    """Records what a fill did to the user's book and tells everyone about it."""

    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, bus: EventBus | None = None
    ):
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.fills = FillRepository(db)
        self.positions = PositionRepository(db)
        self.users = UserRepository(db)
        self.notifications = NotificationService(db, self.settings)

    def publish_order(self, order: Order) -> None:
        self.bus.publish(Topics.ORDER, order_event(order))

    async def apply_position(
        self, order: Order, quantity: float, price: float, fee: float
    ) -> tuple[Position | None, float]:
        """Update the position for a fill. Returns ``(position, realized_pnl)``; the position is
        ``None`` for a sell that exceeds what we track (live mode: coins bought elsewhere)."""
        if order.side == OrderSide.BUY:
            position = await self.positions.apply_buy(order.user_id, order.mode, order.symbol, quantity, price)
            return position, 0.0
        result = await self.positions.apply_sell(order.user_id, order.mode, order.symbol, quantity, price, fee)
        if result is None:
            return None, 0.0
        return result

    async def insert_fill(
        self,
        order: Order,
        quantity: float,
        price: float,
        fee: float,
        realized: float,
        *,
        exchange_trade_id: str | None = None,
    ) -> Fill:
        fill = Fill(
            order_id=order.id or "",
            user_id=order.user_id,
            mode=order.mode,
            symbol=order.symbol,
            side=order.side,
            quantity=round_qty(quantity),
            price=round_price(price),
            fee=round(fee, 8),
            fee_asset=order.fee_asset,
            realized_pnl=round_quote(realized),
            exchange_trade_id=exchange_trade_id,
        )
        return await self.fills.insert(fill)

    async def _wants_order_updates(self, user_id: str) -> bool:
        user = await self.users.get(user_id)
        return bool(user and user.preferences.notifications.order_updates)

    async def notify_fill(self, order: Order, fill: Fill) -> None:
        if not await self._wants_order_updates(order.user_id):
            return
        body = (
            f"{order.type.replace('_', ' ').title()} order filled at {fill.price:,.2f} {order.fee_asset} "
            f"(fee {fill.fee:,.2f} {fill.fee_asset})"
        )
        if order.side == OrderSide.SELL:
            body += f"; realised PnL {fill.realized_pnl:+,.2f} {order.fee_asset}"
        await self._notify(
            order,
            title=f"{order.mode.title()} order filled: {order.side.upper()} {fill.quantity:g} {order.symbol}",
            body=body,
            severity=Severity.INFO,
        )

    async def notify_rejected(self, order: Order) -> None:
        if not await self._wants_order_updates(order.user_id):
            return
        await self._notify(
            order,
            title=f"{order.mode.title()} order rejected: {order.side.upper()} {order.quantity:g} {order.symbol}",
            body=order.reason or "The order could not be executed",
            severity=Severity.WARNING,
        )

    async def _notify(self, order: Order, *, title: str, body: str, severity: Severity) -> None:
        try:
            await self.notifications.notify(
                order.user_id,
                type=NotificationType.ORDER_UPDATE,
                title=title,
                body=body,
                severity=severity,
                data={
                    "order_id": order.id or "",
                    "symbol": order.symbol,
                    "side": str(order.side),
                    "status": str(order.status),
                    "mode": str(order.mode),
                },
            )
        except Exception as exc:  # notifications must never break execution
            logger.warning("order_notification_failed", order_id=order.id, error=str(exc))


class PortfolioService:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, bus: EventBus | None = None
    ):
        self.db = db
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.orders = OrderRepository(db)
        self.fills = FillRepository(db)
        self.positions = PositionRepository(db)
        self.accounts = PaperAccountRepository(db)
        self.snapshots = PortfolioSnapshotRepository(db)
        self.stats = AssetStatsRepository(db)
        self.market = MarketService(db, self.settings)

    # -------------------------------------------------------------- prices
    async def prices(self, symbols: list[str]) -> dict[str, float | None]:
        wanted = sorted({s.upper() for s in symbols})
        if not wanted:
            return {}
        result: dict[str, float | None] = dict.fromkeys(wanted)
        for stats in await self.stats.find({"symbol": {"$in": wanted}}):
            if stats.price is not None:
                result[stats.symbol] = stats.price
        for symbol, price in list(result.items()):
            if price is None:
                result[symbol] = await self.market.latest_price(symbol)
        return result

    # ----------------------------------------------------------- valuation
    async def valuation(
        self,
        user_id: str,
        mode: TradeMode = TradeMode.PAPER,
        *,
        live_cash: float | None = None,
        note: str | None = None,
    ) -> PortfolioOut:
        now = utcnow()
        quote = self.settings.quote_asset
        starting: float | None = None
        high_water: float | None = None
        if mode == TradeMode.PAPER:
            account = await self.accounts.get_or_create(
                user_id, quote_asset=quote, starting_balance=self.settings.paper_starting_balance
            )
            cash = account.cash_balance
            realized = account.realized_pnl
            fees = account.fees_paid
            starting = account.starting_balance
            high_water = account.equity_high_water
            quote = account.quote_asset
        else:
            cash = round_quote(live_cash or 0.0)
            realized, fees = await self._live_totals(user_id)
            if live_cash is None:
                note = note or "Live cash balance unavailable; equity counts tracked positions only"

        positions = await self.positions.open_for_user(user_id, mode)
        prices = await self.prices([p.symbol for p in positions])
        rows: list[PortfolioPosition] = []
        positions_value = 0.0
        cost_basis = 0.0
        for position in positions:
            price = prices.get(position.symbol)
            market_value = round_quote(position.quantity * (price if price is not None else position.avg_entry_price))
            unrealized = round_quote(market_value - position.cost_basis)
            positions_value += market_value
            cost_basis += position.cost_basis
            rows.append(
                PortfolioPosition(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    current_price=price,
                    market_value=market_value,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_pct=round(unrealized / position.cost_basis * 100, 4)
                    if position.cost_basis
                    else None,
                    realized_pnl=position.realized_pnl,
                    allocation_pct=0.0,
                )
            )
        positions_value = round_quote(positions_value)
        equity = round_quote(cash + positions_value)
        unrealized_total = round_quote(sum(r.unrealized_pnl for r in rows))

        allocation: list[AllocationSlice] = []
        for row in rows:
            row.allocation_pct = round(row.market_value / equity * 100, 2) if equity > 0 else 0.0
            allocation.append(AllocationSlice(symbol=row.symbol, pct=row.allocation_pct, value=row.market_value))
        allocation.append(
            AllocationSlice(symbol="CASH", pct=round(cash / equity * 100, 2) if equity > 0 else 0.0, value=cash)
        )
        allocation.sort(key=lambda slice_: -slice_.value)

        day_change = await self._day_change_pct(user_id, mode, equity, now)
        if high_water is not None:
            high_water = max(high_water, equity)
        drawdown = round((high_water - equity) / high_water * 100, 4) if high_water else None

        return PortfolioOut(
            mode=mode,
            quote_asset=quote,
            cash=round_quote(cash),
            positions_value=positions_value,
            equity=equity,
            unrealized_pnl=unrealized_total,
            unrealized_pnl_pct=round(unrealized_total / cost_basis * 100, 4) if cost_basis else None,
            realized_pnl=round_quote(realized),
            fees_paid=round_quote(fees),
            starting_balance=starting,
            total_return_pct=round((equity / starting - 1) * 100, 4) if starting else None,
            day_change_pct=day_change,
            equity_high_water=high_water,
            drawdown_pct=drawdown,
            positions=rows,
            allocation=allocation,
            note=note,
            updated_at=now,
        )

    async def _live_totals(self, user_id: str) -> tuple[float, float]:
        rows = await self.fills.aggregate(
            [
                {"$match": {"user_id": user_id, "mode": str(TradeMode.LIVE)}},
                {"$group": {"_id": None, "realized": {"$sum": "$realized_pnl"}, "fees": {"$sum": "$fee"}}},
            ]
        )
        if not rows:
            return 0.0, 0.0
        return float(rows[0].get("realized") or 0.0), float(rows[0].get("fees") or 0.0)

    async def _day_change_pct(self, user_id: str, mode: TradeMode, equity: float, now: datetime) -> float | None:
        snapshot = await self.snapshots.at_or_before(user_id, mode, now - DAY)
        if not snapshot:
            return None
        base = float(snapshot.get("equity") or 0.0)
        if base <= 0:
            return None
        return round((equity / base - 1) * 100, 4)

    # ----------------------------------------------------------- snapshots
    def portfolio_event(self, user_id: str, portfolio: PortfolioOut) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "mode": str(portfolio.mode),
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "positions_value": portfolio.positions_value,
            "unrealized_pnl": portfolio.unrealized_pnl,
            "realized_pnl": portfolio.realized_pnl,
            "day_change_pct": portfolio.day_change_pct,
            "updated_at": portfolio.updated_at.isoformat(),
        }

    async def publish(self, user_id: str, mode: TradeMode = TradeMode.PAPER) -> PortfolioOut:
        portfolio = await self.valuation(user_id, mode)
        self.bus.publish(Topics.PORTFOLIO, self.portfolio_event(user_id, portfolio))
        return portfolio

    async def snapshot(self, user_id: str, mode: TradeMode = TradeMode.PAPER, *, publish: bool = True) -> PortfolioOut:
        portfolio = await self.valuation(user_id, mode)
        await self.snapshots.insert(
            user_id,
            mode,
            {
                "equity": portfolio.equity,
                "cash": portfolio.cash,
                "positions_value": portfolio.positions_value,
                "unrealized_pnl": portfolio.unrealized_pnl,
                "realized_pnl": portfolio.realized_pnl,
            },
            at=portfolio.updated_at,
        )
        if mode == TradeMode.PAPER:
            await self.accounts.raise_high_water(user_id, portfolio.equity)
        if publish:
            self.bus.publish(Topics.PORTFOLIO, self.portfolio_event(user_id, portfolio))
        return portfolio

    async def snapshot_all(self) -> dict[str, Any]:
        """Snapshot every paper account that holds a position or traded in the last 7 days."""
        candidates = await self.positions.open_user_ids(TradeMode.PAPER)
        candidates |= await self.orders.active_user_ids(mode=TradeMode.PAPER, since=utcnow() - ACTIVE_WINDOW)
        if not candidates:
            return {"snapshots": 0, "errors": 0}
        accounts = await self.accounts.find({"user_id": {"$in": sorted(candidates)}}, projection={"user_id": 1})
        written = 0
        errors = 0
        for user_id in sorted(a.user_id for a in accounts):
            try:
                await self.snapshot(user_id, TradeMode.PAPER)
                written += 1
            except Exception as exc:  # one broken account must not stop the others
                errors += 1
                logger.warning("portfolio_snapshot_failed", user_id=user_id, error=str(exc))
        return {"snapshots": written, "errors": errors}

    async def history(
        self, user_id: str, mode: TradeMode, *, range_: str = "7d", interval: str = "1h"
    ) -> PortfolioHistoryOut:
        if interval not in self.snapshots.BUCKETS:
            raise ValidationFailedError(f"Unsupported interval '{interval}'. Use one of {list(self.snapshots.BUCKETS)}")
        try:
            span = parse_range(range_)
        except ValueError as exc:
            raise ValidationFailedError(f"Invalid range '{range_}' (examples: 24h, 7d, 4w)") from exc
        if span <= timedelta(0) or span > MAX_HISTORY:
            raise ValidationFailedError("Range must be between 1 minute and 366 days")
        rows = await self.snapshots.history(user_id, mode, start=utcnow() - span, interval=interval)
        points = [
            PortfolioPoint(
                ts=row["ts"],
                equity=float(row.get("equity") or 0.0),
                cash=float(row.get("cash") or 0.0),
                positions_value=float(row.get("positions_value") or 0.0),
                unrealized_pnl=float(row.get("unrealized_pnl") or 0.0),
                realized_pnl=float(row.get("realized_pnl") or 0.0),
                high=float(row.get("high") or 0.0),
                low=float(row.get("low") or 0.0),
                samples=int(row.get("samples") or 0),
            )
            for row in rows
        ]
        return PortfolioHistoryOut(mode=mode, range=range_, interval=interval, points=points, count=len(points))


__all__ = ["ACTIVE_WINDOW", "DAY", "Ledger", "PortfolioService", "order_event"]
