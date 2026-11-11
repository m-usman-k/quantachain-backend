"""CCXT adapter for live order execution (FE-1 / FE-3).

One adapter instance wraps one authenticated ``ccxt.async_support`` exchange.
Our order types map onto ccxt's unified API; exchanges that reject the unified
``stopLossPrice`` parameter (notably Binance spot) fall back to their native
stop-limit syntax. Every ccxt error is translated into an ``AppError`` so the
API never leaks exchange internals.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.core.exceptions import ExternalServiceError, UnauthorizedError, ValidationFailedError
from app.modules.trading.models import Order, OrderStatus, OrderType

logger = structlog.get_logger(__name__)

SUPPORTED_EXCHANGES: tuple[str, ...] = ("binance", "kraken", "coinbase", "okx", "bybit", "kucoin", "bitget", "gate")
STOP_LIMIT_OFFSET = 0.005  # native stop-limit fallback: limit price 0.5% beyond the stop

CCXT_STATUS_MAP: dict[str, OrderStatus] = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}


def _ccxt() -> Any:
    import ccxt.async_support as ccxt_async

    return ccxt_async


def translate_error(exc: Exception, exchange_id: str) -> Exception:
    """Map ccxt exception classes to the API error model."""
    try:
        import ccxt

        if isinstance(exc, ccxt.AuthenticationError | ccxt.PermissionDenied):
            return UnauthorizedError(f"{exchange_id}: API key rejected ({exc.__class__.__name__})")
        if isinstance(exc, ccxt.InsufficientFunds | ccxt.InvalidOrder | ccxt.BadSymbol | ccxt.BadRequest):
            return ValidationFailedError(f"{exchange_id}: {str(exc)[:200]}")
        if isinstance(exc, ccxt.NetworkError | ccxt.ExchangeError):
            return ExternalServiceError(f"{exchange_id}: {exc.__class__.__name__}", details={"message": str(exc)[:300]})
    except ImportError:  # pragma: no cover - ccxt is a hard dependency
        if any(name in exc.__class__.__name__.lower() for name in ("invalidorder", "badrequest", "insufficientfunds")):
            return ValidationFailedError(f"{exchange_id}: {str(exc)[:200]}")
    return ExternalServiceError(f"{exchange_id}: {exc.__class__.__name__}", details={"message": str(exc)[:300]})


class CcxtAdapter:
    def __init__(
        self,
        exchange_id: str,
        *,
        api_key: str | None = None,
        secret: str | None = None,
        passphrase: str | None = None,
        testnet: bool = False,
        exchange: Any | None = None,
    ) -> None:
        self.exchange_id = exchange_id.lower()
        self.testnet = testnet
        if exchange is not None:
            self.exchange = exchange
        else:
            self.exchange = self._build(api_key, secret, passphrase)

    def _build(self, api_key: str | None, secret: str | None, passphrase: str | None) -> Any:
        ccxt_async = _ccxt()
        cls = getattr(ccxt_async, self.exchange_id, None)
        if cls is None:
            raise ValidationFailedError(f"Unsupported exchange '{self.exchange_id}'")
        config: dict[str, Any] = {"enableRateLimit": True}
        if api_key:
            config["apiKey"] = api_key
        if secret:
            config["secret"] = secret
        if passphrase:
            config["password"] = passphrase
        exchange = cls(config)
        if self.testnet:
            try:
                exchange.set_sandbox_mode(True)
            except Exception as exc:  # exchange without a sandbox
                raise ValidationFailedError(f"{self.exchange_id} has no testnet/sandbox mode") from exc
        return exchange

    async def close(self) -> None:
        close = getattr(self.exchange, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # pragma: no cover - best effort
                pass

    async def __aenter__(self) -> CcxtAdapter:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------- calls
    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return await getattr(self.exchange, method)(*args, **kwargs)
        except Exception as exc:
            raise translate_error(exc, self.exchange_id) from exc

    async def verify(self) -> dict[str, Any]:
        """Prove the key works by fetching balances; returns non-zero balances and inferred permissions."""
        balance = await self._call("fetch_balance")
        totals = balance.get("total", {}) if isinstance(balance, dict) else {}
        nonzero = {asset: float(amount) for asset, amount in totals.items() if amount and float(amount) > 0}
        permissions = ["read"]
        info = balance.get("info", {}) if isinstance(balance, dict) else {}
        if isinstance(info, dict) and info.get("canTrade", True):
            permissions.append("trade")
        return {"balances": nonzero, "permissions": permissions}

    async def fetch_balance(self) -> dict[str, float]:
        balance = await self._call("fetch_balance")
        totals = balance.get("free", {}) if isinstance(balance, dict) else {}
        return {asset: float(amount) for asset, amount in totals.items() if amount is not None}

    async def create_order(self, order: Order) -> dict[str, Any]:
        """Submit ``order``; returns the parsed exchange order."""
        side = str(order.side)
        if order.type == OrderType.MARKET:
            raw = await self._call("create_order", order.pair, "market", side, order.quantity)
        elif order.type == OrderType.LIMIT:
            raw = await self._call("create_order", order.pair, "limit", side, order.quantity, order.price)
        else:
            raw = await self._create_stop(order, side)
        return self.parse_order(raw)

    async def _create_stop(self, order: Order, side: str) -> Any:
        params = {"stopLossPrice": order.stop_price}
        try:
            if order.type == OrderType.STOP_LOSS:
                return await self._call("create_order", order.pair, "market", side, order.quantity, None, params)
            return await self._call("create_order", order.pair, "limit", side, order.quantity, order.price, params)
        except ValidationFailedError as exc:
            if self.exchange_id != "binance":
                raise
            logger.info("binance_stop_fallback", reason=str(exc))
        stop = float(order.stop_price or 0.0)
        limit = (
            order.price
            if order.type == OrderType.STOP_LIMIT and order.price
            else stop * (1 - STOP_LIMIT_OFFSET if side == "sell" else 1 + STOP_LIMIT_OFFSET)
        )
        native = {"stopPrice": stop, "type": "STOP_LOSS_LIMIT", "timeInForce": "GTC"}
        return await self._call("create_order", order.pair, "limit", side, order.quantity, limit, native)

    async def cancel_order(self, exchange_order_id: str, pair: str) -> dict[str, Any]:
        raw = await self._call("cancel_order", exchange_order_id, pair)
        return (
            self.parse_order(raw)
            if isinstance(raw, dict) and raw.get("id")
            else {"id": exchange_order_id, "status": OrderStatus.CANCELED}
        )

    async def fetch_order(self, exchange_order_id: str, pair: str) -> dict[str, Any]:
        return self.parse_order(await self._call("fetch_order", exchange_order_id, pair))

    async def fetch_open_orders(self, pair: str | None = None) -> list[dict[str, Any]]:
        rows = await self._call("fetch_open_orders", pair) if pair else await self._call("fetch_open_orders")
        return [self.parse_order(r) for r in rows or []]

    # ------------------------------------------------------------ parsing
    @staticmethod
    def parse_order(raw: dict[str, Any]) -> dict[str, Any]:
        """Normalise a ccxt order structure into the subset the service needs."""
        status_text = str(raw.get("status") or "open").lower()
        filled = float(raw.get("filled") or 0.0)
        amount = float(raw.get("amount") or 0.0)
        status = CCXT_STATUS_MAP.get(status_text, OrderStatus.OPEN)
        if status == OrderStatus.OPEN and 0 < filled < amount:
            status = OrderStatus.PARTIALLY_FILLED
        fee = raw.get("fee") or {}
        return {
            "id": str(raw.get("id") or ""),
            "status": status,
            "filled": filled,
            "amount": amount,
            "average": float(raw["average"])
            if raw.get("average") is not None
            else (float(raw["price"]) if raw.get("price") is not None else None),
            "fee": float(fee.get("cost") or 0.0) if isinstance(fee, dict) else 0.0,
            "fee_asset": (fee.get("currency") if isinstance(fee, dict) else None) or "USDT",
            "timestamp": raw.get("timestamp"),
            "raw_status": status_text,
        }


__all__ = ["CCXT_STATUS_MAP", "STOP_LIMIT_OFFSET", "SUPPORTED_EXCHANGES", "CcxtAdapter", "translate_error"]
