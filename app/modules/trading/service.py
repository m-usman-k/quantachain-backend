"""Module 9 - Trading & Order Execution service (paper by default, CCXT live when enabled)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings, get_settings
from app.core.events import EventBus, event_bus
from app.core.exceptions import AppError, ConflictError, FeatureDisabledError, NotFoundError, ValidationFailedError
from app.core.pagination import PageParams
from app.core.security import SecretBox
from app.core.timeutils import utcnow
from app.integrations.binance import BinanceClient
from app.modules.auth.models import User
from app.modules.market.repository import AssetRepository
from app.modules.market.service import MarketService
from app.modules.trading.exchange import SUPPORTED_EXCHANGES, CcxtAdapter
from app.modules.trading.models import (
    LIMIT_TYPES,
    STOP_TYPES,
    WORKING_STATUSES,
    ExchangeCredential,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    TradeMode,
    round_price,
    round_qty,
)
from app.modules.trading.paper import SLIPPAGE_BPS, PaperEngine
from app.modules.trading.portfolio import Ledger, PortfolioService
from app.modules.trading.repository import (
    ExchangeCredentialRepository,
    FillRepository,
    OrderRepository,
    PaperAccountRepository,
    PositionRepository,
)
from app.modules.trading.schemas import (
    ExchangeKeyCreate,
    OrderBookOut,
    OrderCreate,
    PaperAccountOut,
    PaperResetOut,
    PortfolioHistoryOut,
    PortfolioOut,
    PositionOut,
    TradingCapabilities,
    TradingSummary,
)

logger = structlog.get_logger(__name__)

PRICE_SANITY_PCT = 50.0  # limit / stop prices more than this far from the market are rejected


class TradingService:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, *, bus: EventBus | None = None
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.orders = OrderRepository(db)
        self.fills = FillRepository(db)
        self.positions = PositionRepository(db)
        self.accounts = PaperAccountRepository(db)
        self.credentials = ExchangeCredentialRepository(db)
        self.assets = AssetRepository(db)
        self.market = MarketService(db, self.settings)
        self.ledger = Ledger(db, self.settings, self.bus)
        self.portfolio = PortfolioService(db, self.settings, self.bus)
        self.engine = PaperEngine(db, self.settings, bus=self.bus, ledger=self.ledger, portfolio=self.portfolio)
        self.secrets = SecretBox(self.settings.fernet_key)

    # ------------------------------------------------------------ capabilities
    def capabilities(self) -> TradingCapabilities:
        return TradingCapabilities(
            live_enabled=self.settings.live_trading_enabled,
            supported_exchanges=list(SUPPORTED_EXCHANGES),
            default_exchange=self.settings.default_exchange,
            quote_asset=self.settings.quote_asset,
            paper_starting_balance=self.settings.paper_starting_balance,
            paper_fee_rate=self.settings.paper_fee_rate,
            paper_slippage_bps=SLIPPAGE_BPS,
            order_types=list(OrderType),
        )

    async def summary(self, user: User) -> TradingSummary:
        user_id = user.id or ""
        now = utcnow()
        portfolio = await self.portfolio.valuation(user_id, TradeMode.PAPER)
        return TradingSummary(
            open_orders=await self.orders.count(
                {"user_id": user_id, "status": {"$in": [str(s) for s in WORKING_STATUSES]}}
            ),
            filled_today=await self.orders.count(
                {
                    "user_id": user_id,
                    "status": str(OrderStatus.FILLED),
                    "filled_at": {"$gte": now - timedelta(hours=24)},
                }
            ),
            paper_equity=portfolio.equity,
            paper_total_return_pct=portfolio.total_return_pct,
            paper_positions=len(portfolio.positions),
            exchange_keys=await self.credentials.count({"user_id": user_id}),
            live_enabled=self.settings.live_trading_enabled,
            generated_at=now,
        )

    # ------------------------------------------------------------------ orders
    async def create_order(self, user: User, payload: OrderCreate) -> Order:
        user_id = user.id or ""
        symbol = payload.symbol
        if await self.assets.get_by_symbol(symbol) is None:
            raise ValidationFailedError(f"{symbol} is not a tracked asset")
        if payload.type in LIMIT_TYPES and payload.price is None:
            raise ValidationFailedError(f"{payload.type} orders need a price")
        if payload.type in STOP_TYPES and payload.stop_price is None:
            raise ValidationFailedError(f"{payload.type} orders need a stop_price")
        if payload.type == OrderType.MARKET and (payload.price is not None or payload.stop_price is not None):
            raise ValidationFailedError("Market orders take neither price nor stop_price")
        exchange = payload.exchange or self.settings.default_exchange
        if payload.mode == TradeMode.LIVE:
            if not self.settings.live_trading_enabled:
                raise FeatureDisabledError("Live trading is disabled on this server (LIVE_TRADING_ENABLED=false)")
            if await self.credentials.get_for_exchange(user_id, exchange) is None:
                raise ValidationFailedError(f"Store API keys for {exchange} before placing live orders")
        last_price = await self.market.latest_price(symbol)
        for label, value in (("price", payload.price), ("stop_price", payload.stop_price)):
            if value is not None and last_price and abs(value / last_price - 1) * 100 > PRICE_SANITY_PCT:
                raise ValidationFailedError(
                    f"{label} {value} is more than {PRICE_SANITY_PCT:.0f}% away from the market price {last_price:,.2f}"
                )
        if payload.client_order_id and await self.orders.exists({"client_order_id": payload.client_order_id}):
            raise ConflictError("An order with this client_order_id already exists")

        order = Order(
            user_id=user_id,
            mode=payload.mode,
            exchange=exchange,
            symbol=symbol,
            pair=f"{symbol}/{self.settings.quote_asset}",
            side=payload.side,
            type=payload.type,
            quantity=round_qty(payload.quantity),
            price=round_price(payload.price) if payload.price is not None else None,
            stop_price=round_price(payload.stop_price) if payload.stop_price is not None else None,
            fee_asset=self.settings.quote_asset,
            source=payload.source,
            **({"client_order_id": payload.client_order_id} if payload.client_order_id else {}),
        )
        try:
            order = await self.orders.insert(order)
        except DuplicateKeyError as exc:
            raise ConflictError("An order with this client_order_id already exists") from exc
        self.ledger.publish_order(order)
        if order.mode == TradeMode.PAPER:
            return await self.engine.place(order, last_price)
        return await LiveExecutor(self).submit(order)

    async def list_orders(
        self, user: User, params: PageParams, *, mode: TradeMode | None, status: OrderStatus | None, symbol: str | None
    ) -> tuple[list[Order], int]:
        query: dict[str, Any] = {"user_id": user.id}
        if mode:
            query["mode"] = str(mode)
        if status:
            query["status"] = str(status)
        if symbol:
            query["symbol"] = symbol.upper()
        return await self.orders.paginate(query, params, sort=[("created_at", -1)])

    async def get_order(self, user: User, order_id: str) -> Order:
        order = await self.orders.get_for_user(user.id or "", order_id)
        if order is None:
            raise NotFoundError("Order not found")
        return order

    async def cancel_order(self, user: User, order_id: str) -> Order:
        order = await self.get_order(user, order_id)
        if order.is_final:
            raise ConflictError(f"Order is already {order.status}")
        if order.mode == TradeMode.PAPER:
            return await self.engine.cancel(order)
        return await LiveExecutor(self).cancel(order)

    async def list_fills(
        self, user: User, params: PageParams, *, mode: TradeMode | None, symbol: str | None
    ) -> tuple[list[Fill], int]:
        query: dict[str, Any] = {"user_id": user.id}
        if mode:
            query["mode"] = str(mode)
        if symbol:
            query["symbol"] = symbol.upper()
        return await self.fills.paginate(query, params, sort=[("created_at", -1)])

    # --------------------------------------------------------------- portfolio
    async def positions_out(self, user: User, mode: TradeMode) -> list[PositionOut]:
        positions = await self.positions.open_for_user(user.id or "", mode)
        prices = await self.portfolio.prices([p.symbol for p in positions])
        return [PositionOut.from_model(p, prices.get(p.symbol)) for p in positions]

    async def portfolio_out(self, user: User, mode: TradeMode) -> PortfolioOut:
        if mode == TradeMode.LIVE:
            cash = await self._live_cash(user)
            return await self.portfolio.valuation(user.id or "", TradeMode.LIVE, live_cash=cash)
        return await self.portfolio.valuation(user.id or "", TradeMode.PAPER)

    async def _live_cash(self, user: User) -> float | None:
        if not self.settings.live_trading_enabled:
            return None
        credential = await self.credentials.get_for_exchange(user.id or "", self.settings.default_exchange)
        if credential is None:
            return None
        try:
            async with self._adapter(credential) as adapter:
                balances = await adapter.fetch_balance()
            return balances.get(self.settings.quote_asset)
        except AppError as exc:
            logger.warning("live_balance_unavailable", user_id=user.id, error=exc.message)
            return None

    async def portfolio_history(
        self, user: User, mode: TradeMode, *, range_: str, interval: str
    ) -> PortfolioHistoryOut:
        return await self.portfolio.history(user.id or "", mode, range_=range_, interval=interval)

    async def paper_reset(self, user: User) -> PaperResetOut:
        account, canceled, closed = await self.engine.reset(user.id or "")
        portfolio = await self.portfolio.snapshot(user.id or "", TradeMode.PAPER)
        return PaperResetOut(
            account=PaperAccountOut.from_model(account),
            canceled_orders=canceled,
            closed_positions=closed,
            portfolio=portfolio,
        )

    async def orderbook(self, symbol: str, *, limit: int = 20) -> OrderBookOut:
        try:
            async with BinanceClient(self.settings) as client:
                book = await client.fetch_order_book(symbol, limit=limit)
            bids, asks = book["bids"], book["asks"]
            last_update_id = book.get("last_update_id")
        except Exception as exc:
            logger.info("binance_orderbook_fallback", symbol=symbol, error=str(exc))
            latest_price = await self.market.latest_price(symbol) or 50000.0
            bids = [
                [round(latest_price * (1.0 - 0.0005 * i), 2), round(0.5 * (1 + 0.1 * i), 4)]
                for i in range(1, limit + 1)
            ]
            asks = [
                [round(latest_price * (1.0 + 0.0005 * i), 2), round(0.5 * (1 + 0.1 * i), 4)]
                for i in range(1, limit + 1)
            ]
            last_update_id = int(utcnow().timestamp() * 1000)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
        return OrderBookOut(
            symbol=symbol.upper(),
            pair=f"{symbol.upper()}/{self.settings.quote_asset}",
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=round(mid, 8) if mid is not None else None,
            spread=round(spread, 8) if spread is not None else None,
            spread_pct=round(spread / mid * 100, 6) if spread is not None and mid else None,
            last_update_id=last_update_id,
            fetched_at=utcnow(),
        )

    # ------------------------------------------------------------ exchange keys
    def _adapter(self, credential: ExchangeCredential) -> CcxtAdapter:
        return CcxtAdapter(
            credential.exchange,
            api_key=self.secrets.decrypt(credential.api_key_encrypted),
            secret=self.secrets.decrypt(credential.secret_encrypted),
            passphrase=self.secrets.decrypt(credential.passphrase_encrypted)
            if credential.passphrase_encrypted
            else None,
            testnet=credential.testnet,
        )

    async def list_keys(self, user: User) -> list[ExchangeCredential]:
        return await self.credentials.for_user(user.id or "")

    async def add_key(self, user: User, payload: ExchangeKeyCreate) -> ExchangeCredential:
        if payload.exchange not in SUPPORTED_EXCHANGES:
            raise ValidationFailedError(
                f"Unsupported exchange '{payload.exchange}'. Supported: {', '.join(SUPPORTED_EXCHANGES)}"
            )
        if await self.credentials.get_for_exchange(user.id or "", payload.exchange):
            raise ConflictError(f"Keys for {payload.exchange} already exist; delete them first")
        credential = ExchangeCredential(
            user_id=user.id or "",
            exchange=payload.exchange,
            label=payload.label or payload.exchange.title(),
            api_key_encrypted=self.secrets.encrypt(payload.api_key),
            secret_encrypted=self.secrets.encrypt(payload.secret),
            passphrase_encrypted=self.secrets.encrypt(payload.passphrase) if payload.passphrase else None,
            key_preview=f"{payload.api_key[:4]}…{payload.api_key[-4:]}",
            testnet=payload.testnet,
        )
        try:
            credential = await self.credentials.insert(credential)
        except DuplicateKeyError as exc:
            raise ConflictError(f"Keys for {payload.exchange} already exist") from exc
        if self.settings.live_trading_enabled:
            credential = await self.verify_key(user, credential.id or "")
        return credential

    async def verify_key(self, user: User, credential_id: str) -> ExchangeCredential:
        credential = await self.credentials.get_for_user(user.id or "", credential_id)
        if credential is None:
            raise NotFoundError("Exchange key not found")
        if not self.settings.live_trading_enabled:
            raise FeatureDisabledError("Live trading is disabled; keys cannot be verified against the exchange")
        try:
            async with self._adapter(credential) as adapter:
                result = await adapter.verify()
            fields = {
                "is_valid": True,
                "permissions": result["permissions"],
                "last_verified_at": utcnow(),
                "last_error": None,
            }
        except AppError as exc:
            fields = {"is_valid": False, "last_verified_at": utcnow(), "last_error": exc.message}
        return await self.credentials.update(credential_id, fields) or credential

    async def delete_key(self, user: User, credential_id: str) -> None:
        credential = await self.credentials.get_for_user(user.id or "", credential_id)
        if credential is None:
            raise NotFoundError("Exchange key not found")
        await self.credentials.delete(credential_id)


class LiveExecutor:
    """Submits and synchronises live orders through the user's stored exchange keys."""

    def __init__(self, service: TradingService) -> None:
        self.service = service

    async def _adapter_for(self, order: Order) -> CcxtAdapter:
        credential = await self.service.credentials.get_for_exchange(order.user_id, order.exchange)
        if credential is None:
            raise ValidationFailedError(f"No API keys stored for {order.exchange}")
        return self.service._adapter(credential)

    async def submit(self, order: Order) -> Order:
        try:
            async with await self._adapter_for(order) as adapter:
                parsed = await adapter.create_order(order)
        except AppError as exc:
            failed = await self.service.orders.transition(
                order.id or "",
                from_statuses=WORKING_STATUSES,
                to_status=OrderStatus.FAILED,
                fields={"reason": exc.message},
            )
            if failed:
                self.service.ledger.publish_order(failed)
            raise
        return await self.apply(order, parsed)

    async def apply(self, order: Order, parsed: dict[str, Any]) -> Order:
        """Reconcile a live order document with the exchange's view, booking new fills."""
        new_filled = round_qty(float(parsed.get("filled") or 0.0))
        delta = round_qty(new_filled - order.filled_quantity)
        fields: dict[str, Any] = {
            "exchange_order_id": parsed.get("id") or order.exchange_order_id,
            "filled_quantity": new_filled,
        }
        if parsed.get("average") is not None:
            fields["avg_fill_price"] = round_price(float(parsed["average"]))
        if parsed.get("fee"):
            fields["fee"] = float(parsed["fee"])
            fields["fee_asset"] = parsed.get("fee_asset") or order.fee_asset
        status = parsed.get("status", OrderStatus.OPEN)
        if status == OrderStatus.FILLED:
            fields["filled_at"] = utcnow()
        updated = await self.service.orders.transition(
            order.id or "", from_statuses=WORKING_STATUSES, to_status=status, fields=fields
        )
        if updated is None:
            return await self.service.orders.get(order.id or "") or order
        if delta > 0 and updated.avg_fill_price:
            fee = float(parsed.get("fee") or 0.0)
            _, realized = await self.service.ledger.apply_position(updated, delta, updated.avg_fill_price, fee)
            fill = await self.service.ledger.insert_fill(
                updated, delta, updated.avg_fill_price, fee, realized, exchange_trade_id=parsed.get("id")
            )
            if status == OrderStatus.FILLED:
                await self.service.ledger.notify_fill(updated, fill)
        self.service.ledger.publish_order(updated)
        return updated

    async def cancel(self, order: Order) -> Order:
        if order.exchange_order_id:
            async with await self._adapter_for(order) as adapter:
                await adapter.cancel_order(order.exchange_order_id, order.pair)
        canceled = await self.service.orders.transition(
            order.id or "",
            from_statuses=WORKING_STATUSES,
            to_status=OrderStatus.CANCELED,
            fields={"reason": "Canceled by user"},
        )
        if canceled is None:
            raise ConflictError("Order is no longer open")
        self.service.ledger.publish_order(canceled)
        return canceled

    async def sync(self) -> dict[str, int]:
        synced = errors = 0
        for order in await self.service.orders.working(mode=TradeMode.LIVE):
            if not order.exchange_order_id:
                continue
            try:
                async with await self._adapter_for(order) as adapter:
                    parsed = await adapter.fetch_order(order.exchange_order_id, order.pair)
                await self.apply(order, parsed)
                synced += 1
            except AppError as exc:
                errors += 1
                logger.warning("live_order_sync_failed", order_id=order.id, error=exc.message)
        return {"synced": synced, "errors": errors}


__all__ = ["PRICE_SANITY_PCT", "LiveExecutor", "TradingService"]
