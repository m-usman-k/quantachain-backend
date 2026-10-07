"""Binance public market data: REST (klines, 24h tickers, order book) and WebSocket streams."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
import websockets

from app.core.config import Settings, get_settings
from app.core.metrics import metrics
from app.core.timeutils import from_ms
from app.integrations.base import ProviderClient

logger = structlog.get_logger(__name__)

# Binance interval codes match ours for the intervals we support.
BINANCE_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}


@dataclass(slots=True)
class Kline:
    symbol: str  # base asset, e.g. BTC
    pair: str  # exchange pair, e.g. BTCUSDT
    interval: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    closed: bool


@dataclass(slots=True)
class Tick:
    symbol: str
    pair: str
    price: float
    ts: datetime
    volume_24h: float | None = None
    change_24h_pct: float | None = None


def split_pair(pair: str, quote: str) -> str:
    pair = pair.upper()
    return pair[: -len(quote)] if pair.endswith(quote.upper()) else pair


class BinanceClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__("binance", settings.binance_rest_url, timeout=15.0)
        self.quote = settings.quote_asset

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Kline]:
        if interval not in BINANCE_INTERVALS:
            raise ValueError(f"Binance does not support interval {interval}")
        pair = f"{symbol.upper()}{self.quote}"
        params: dict[str, Any] = {"symbol": pair, "interval": interval, "limit": min(limit, 1000)}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        rows = await self.get_json("/api/v3/klines", params=params)
        klines: list[Kline] = []
        for row in rows:
            klines.append(
                Kline(
                    symbol=symbol.upper(),
                    pair=pair,
                    interval=interval,
                    open_time=from_ms(row[0]),
                    close_time=from_ms(row[6]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[7]),
                    trades=int(row[8]),
                    closed=True,
                )
            )
        if klines:
            # The last kline returned without an end time is usually still open.
            now_ms = int(datetime.now().timestamp() * 1000)
            if klines[-1].close_time.timestamp() * 1000 > now_ms:
                klines[-1].closed = False
        return klines

    async def fetch_24h_tickers(self, symbols: list[str]) -> list[Tick]:
        pairs = [f"{s.upper()}{self.quote}" for s in symbols]
        rows = await self.get_json("/api/v3/ticker/24hr", params={"symbols": json.dumps(pairs, separators=(",", ":"))})
        ticks = []
        for row in rows:
            ticks.append(
                Tick(
                    symbol=split_pair(row["symbol"], self.quote),
                    pair=row["symbol"],
                    price=float(row["lastPrice"]),
                    ts=from_ms(row["closeTime"]),
                    volume_24h=float(row["quoteVolume"]),
                    change_24h_pct=float(row["priceChangePercent"]),
                )
            )
        return ticks

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        pair = f"{symbol.upper()}{self.quote}"
        data = await self.get_json("/api/v3/depth", params={"symbol": pair, "limit": limit})
        return {
            "symbol": symbol.upper(),
            "pair": pair,
            "bids": [[float(p), float(q)] for p, q in data["bids"]],
            "asks": [[float(p), float(q)] for p, q in data["asks"]],
            "last_update_id": data.get("lastUpdateId"),
        }


class BinanceStream:
    """Combined miniTicker + 1m kline stream with automatic reconnection."""

    def __init__(self, symbols: list[str], settings: Settings | None = None, *, kline_interval: str = "1m") -> None:
        self.settings = settings or get_settings()
        self.symbols = [s.upper() for s in symbols]
        self.kline_interval = kline_interval
        self.stats = metrics.provider("binance_ws")
        self.reconnects = 0
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _url(self) -> str:
        streams = []
        for symbol in self.symbols:
            pair = f"{symbol}{self.settings.quote_asset}".lower()
            streams.append(f"{pair}@miniTicker")
            streams.append(f"{pair}@kline_{self.kline_interval}")
        return f"{self.settings.binance_ws_url}?streams={'/'.join(streams)}"

    async def __aiter__(self) -> AsyncIterator[Tick | Kline]:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self._url(), ping_interval=20, ping_timeout=20, max_queue=2000) as ws:
                    logger.info("binance_ws_connected", symbols=len(self.symbols))
                    backoff = 1.0
                    async for raw in ws:
                        if self._stop.is_set():
                            return
                        parsed = self._parse(raw)
                        if parsed is not None:
                            self.stats.record(0.0)
                            yield parsed
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # websockets raises many connection-related errors
                if self._stop.is_set():
                    return
                self.reconnects += 1
                self.stats.record(0.0, error=f"{type(exc).__name__}: {exc}")
                logger.warning("binance_ws_disconnected", error=str(exc), retry_in=round(backoff, 1))
                await asyncio.sleep(backoff + random.random())
                backoff = min(backoff * 2, 60.0)

    def _parse(self, raw: str | bytes) -> Tick | Kline | None:
        try:
            message = json.loads(raw)
        except ValueError:
            return None
        data = message.get("data") or message
        event = data.get("e")
        quote = self.settings.quote_asset
        if event == "24hrMiniTicker":
            return Tick(
                symbol=split_pair(data["s"], quote),
                pair=data["s"],
                price=float(data["c"]),
                ts=from_ms(data["E"]),
                volume_24h=float(data["q"]),
                change_24h_pct=((float(data["c"]) / float(data["o"])) - 1) * 100 if float(data["o"]) else None,
            )
        if event == "kline":
            k = data["k"]
            return Kline(
                symbol=split_pair(k["s"], quote),
                pair=k["s"],
                interval=k["i"],
                open_time=from_ms(k["t"]),
                close_time=from_ms(k["T"]),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                quote_volume=float(k["q"]),
                trades=int(k["n"]),
                closed=bool(k["x"]),
            )
        return None


__all__ = ["BINANCE_INTERVALS", "BinanceClient", "BinanceStream", "Kline", "Tick", "split_pair"]
