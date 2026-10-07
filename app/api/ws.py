"""WebSocket endpoints.

* ``/ws/market``     - live ticks & candles (public). Query ``?symbols=BTC,ETH`` or send
                       ``{"action": "subscribe", "symbols": [...]}``.
* ``/ws/onchain``    - whale transfer feed (public).
* ``/ws/alerts``     - per-user notifications, fraud alerts and trade signals (auth).
* ``/ws/portfolio``  - order fills and portfolio PnL updates (auth).
* ``/ws/ingestion``  - live ingestion log lines and source status (admin).

Authenticate with ``?token=<access token>`` or an ``Authorization: Bearer`` header.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import structlog
from bson import ObjectId
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.events import Event, Subscription, Topics, event_bus
from app.core.metrics import metrics
from app.core.security import TokenError, decode_token
from app.db.ingestion_logs import serialise_log, tail_ingestion_logs
from app.modules.auth.models import Role, User
from app.modules.auth.repository import UserRepository

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["websocket"])

HEARTBEAT_SECONDS = 25.0


class WsHub:
    """Background plumbing shared by all sockets (currently: ingestion log tailing)."""

    def __init__(self) -> None:
        self._db: AsyncDatabase[dict[str, Any]] | None = None
        self._settings: Settings | None = None
        self._tail_task: asyncio.Task[None] | None = None
        self.ingestion_listeners = 0

    async def start(self, db: AsyncDatabase[dict[str, Any]], settings: Settings) -> None:
        self._db = db
        self._settings = settings

    async def stop(self) -> None:
        if self._tail_task:
            self._tail_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tail_task
            self._tail_task = None

    @property
    def db(self) -> AsyncDatabase[dict[str, Any]]:
        if self._db is None:
            raise RuntimeError("WsHub is not started")
        return self._db

    def ensure_log_tailer(self) -> None:
        """Replay worker-process ingestion logs onto the local event bus."""
        if self._tail_task is None or self._tail_task.done():
            self._tail_task = asyncio.create_task(self._tail_logs(), name="ws-ingestion-tailer")

    async def _tail_logs(self) -> None:
        last_id: ObjectId | None = None
        settings = self._settings
        same_process = bool(settings and settings.run_workers_in_api)
        try:
            while self.ingestion_listeners > 0:
                try:
                    rows = await tail_ingestion_logs(self.db, after_id=last_id, limit=200)
                    if rows:
                        last_id = rows[-1]["_id"]
                        # When the worker runs inside this process the bus already carries the lines.
                        if not same_process:
                            for row in rows:
                                event_bus.publish(Topics.INGESTION_LOG, serialise_log(row))
                except Exception as exc:  # pragma: no cover - keep tailing on transient errors
                    logger.warning("ingestion_tail_failed", error=str(exc))
                await asyncio.sleep(1.0)
        finally:
            self._tail_task = None


ws_hub = WsHub()


# ----------------------------------------------------------------- helpers
async def _authenticate(websocket: WebSocket, token: str | None) -> User | None:
    if not token:
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:]
    if not token:
        return None
    try:
        payload = decode_token(token, "access")
    except TokenError:
        return None
    user = await UserRepository(ws_hub.db).get(payload["sub"])
    if user is None or not user.is_active or user.token_version != payload.get("ver", 0):
        return None
    return user


async def _send(websocket: WebSocket, message: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(message, default=str))


async def _pump(
    websocket: WebSocket,
    subscription: Subscription,
    *,
    accept: Any,
    on_client_message: Any = None,
) -> None:
    """Forward bus events to the socket while handling client control messages."""
    metrics.websocket_connections += 1
    receiver = asyncio.create_task(_receive_loop(websocket, on_client_message))
    try:
        while True:
            if receiver.done():
                break
            event = await subscription.get(timeout=HEARTBEAT_SECONDS)
            if event is None:
                await _send(websocket, {"type": "ping", "dropped": subscription.dropped})
                continue
            if accept(event):
                await _send(
                    websocket,
                    {
                        "type": "event",
                        "topic": event.topic,
                        "data": event.payload,
                        "ts": event.published_at.isoformat(),
                    },
                )
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        metrics.websocket_connections -= 1
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await receiver
        subscription.close()


async def _receive_loop(websocket: WebSocket, handler: Any) -> None:
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except ValueError:
                await _send(websocket, {"type": "error", "message": "Messages must be JSON"})
                continue
            if message.get("action") == "ping":
                await _send(websocket, {"type": "pong"})
            elif handler is not None:
                await handler(message)
    except WebSocketDisconnect:
        return


# --------------------------------------------------------------- endpoints
@router.websocket("/ws/market")
async def ws_market(websocket: WebSocket, symbols: str | None = Query(default=None)) -> None:
    await websocket.accept()
    wanted: set[str] = {s.strip().upper() for s in (symbols or "").split(",") if s.strip()}

    async def handle(message: dict[str, Any]) -> None:
        action = message.get("action")
        requested = {str(s).upper() for s in message.get("symbols", [])}
        if action == "subscribe":
            wanted.update(requested)
        elif action == "unsubscribe":
            wanted.difference_update(requested)
        await _send(websocket, {"type": "subscribed", "symbols": sorted(wanted)})

    await _send(websocket, {"type": "subscribed", "symbols": sorted(wanted) or ["*"]})
    subscription = event_bus.subscribe(Topics.PRICE_TICK, Topics.CANDLE, Topics.MARKET_OVERVIEW)

    def accept(event: Event) -> bool:
        if not wanted or event.topic == Topics.MARKET_OVERVIEW:
            return True
        return str(event.payload.get("symbol", "")).upper() in wanted

    await _pump(websocket, subscription, accept=accept, on_client_message=handle)


@router.websocket("/ws/onchain")
async def ws_onchain(websocket: WebSocket, min_usd: float = Query(default=0.0)) -> None:
    await websocket.accept()
    subscription = event_bus.subscribe(Topics.WHALE_TRANSFER)
    await _pump(websocket, subscription, accept=lambda e: float(e.payload.get("amount_usd", 0) or 0) >= min_usd)


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
    user = await _authenticate(websocket, token)
    if user is None:
        await websocket.close(code=4401, reason="Authentication required")
        return
    await websocket.accept()
    await _send(websocket, {"type": "ready", "user_id": user.id})
    subscription = event_bus.subscribe(Topics.ALERT, Topics.FRAUD_ALERT, Topics.SIGNAL, Topics.WHALE_TRANSFER)
    prefs = user.preferences.notifications

    def accept(event: Event) -> bool:
        payload = event.payload
        target = payload.get("user_id")
        if target and target != user.id:
            return False
        if event.topic == Topics.FRAUD_ALERT:
            return prefs.fraud_alerts
        if event.topic == Topics.SIGNAL:
            return prefs.trade_signals
        if event.topic == Topics.WHALE_TRANSFER:
            return prefs.whale_alerts and float(payload.get("amount_usd", 0) or 0) >= prefs.min_whale_usd
        return True

    await _pump(websocket, subscription, accept=accept)


@router.websocket("/ws/portfolio")
async def ws_portfolio(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
    user = await _authenticate(websocket, token)
    if user is None:
        await websocket.close(code=4401, reason="Authentication required")
        return
    await websocket.accept()
    await _send(websocket, {"type": "ready", "user_id": user.id})
    subscription = event_bus.subscribe(Topics.ORDER, Topics.PORTFOLIO)
    await _pump(websocket, subscription, accept=lambda e: e.payload.get("user_id") == user.id)


@router.websocket("/ws/ingestion")
async def ws_ingestion(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
    user = await _authenticate(websocket, token)
    if user is None or user.role != Role.ADMIN:
        await websocket.close(code=4403, reason="Admin access required")
        return
    await websocket.accept()
    ws_hub.ingestion_listeners += 1
    ws_hub.ensure_log_tailer()
    try:
        recent = await tail_ingestion_logs(ws_hub.db, limit=50)
        await _send(websocket, {"type": "history", "lines": [serialise_log(r) for r in recent]})
        subscription = event_bus.subscribe(Topics.INGESTION_LOG, Topics.SYSTEM)
        await _pump(websocket, subscription, accept=lambda e: True)
    finally:
        ws_hub.ingestion_listeners -= 1


__all__ = ["WsHub", "router", "ws_hub"]
