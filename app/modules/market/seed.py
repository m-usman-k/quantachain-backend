"""Default assets and synthetic market data for demos and tests."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.timeutils import floor_time, interval_delta, utcnow
from app.integrations.coingecko import DEFAULT_COINGECKO_IDS
from app.modules.market.models import Candle
from app.modules.market.repository import AssetRepository, AssetStatsRepository, CandleRepository

ASSET_NAMES: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "BNB": "BNB",
    "XRP": "XRP",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "AVAX": "Avalanche",
    "LINK": "Chainlink",
    "DOT": "Polkadot",
    "MATIC": "Polygon",
    "POL": "Polygon",
    "TRX": "TRON",
    "LTC": "Litecoin",
    "UNI": "Uniswap",
    "ATOM": "Cosmos",
    "NEAR": "NEAR Protocol",
    "ARB": "Arbitrum",
    "OP": "Optimism",
    "SUI": "Sui",
    "APT": "Aptos",
    "PEPE": "Pepe",
    "SHIB": "Shiba Inu",
    "USDT": "Tether",
    "USDC": "USD Coin",
    "WBTC": "Wrapped Bitcoin",
}

REFERENCE_PRICES: dict[str, float] = {
    "BTC": 64_000.0,
    "ETH": 3_400.0,
    "SOL": 145.0,
    "BNB": 580.0,
    "XRP": 0.55,
    "ADA": 0.42,
    "DOGE": 0.12,
    "AVAX": 28.0,
    "LINK": 14.5,
    "DOT": 6.2,
}


async def ensure_default_assets(db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None) -> int:
    """Make sure every tracked symbol has an ``assets`` row. Safe to run on each start-up."""
    settings = settings or get_settings()
    repo = AssetRepository(db)
    created = 0
    for rank, symbol in enumerate(settings.tracked_symbols, start=1):
        existing = await repo.get_by_symbol(symbol)
        if existing is None:
            await repo.upsert_asset(
                symbol,
                name=ASSET_NAMES.get(symbol, symbol),
                coingecko_id=DEFAULT_COINGECKO_IDS.get(symbol),
                binance_symbol=f"{symbol}{settings.quote_asset}",
                rank=rank,
            )
            created += 1
    return created


def generate_synthetic_candles(
    symbol: str,
    interval: str,
    count: int,
    *,
    end: datetime | None = None,
    start_price: float | None = None,
    seed: int | None = None,
    volatility: float = 0.012,
) -> list[Candle]:
    """Geometric random walk with a gentle sine trend - good enough to exercise charts and models."""
    rng = random.Random(seed if seed is not None else hash(symbol) & 0xFFFF)
    step = interval_delta(interval)
    end = floor_time(end or utcnow(), interval)
    price = start_price or REFERENCE_PRICES.get(symbol.upper(), 100.0)
    candles: list[Candle] = []
    open_time = end - step * (count - 1)
    for i in range(count):
        drift = 0.0004 * math.sin(i / max(count / 6, 1))
        change = rng.gauss(drift, volatility)
        open_price = price
        close_price = max(open_price * (1 + change), 1e-6)
        high = max(open_price, close_price) * (1 + abs(rng.gauss(0, volatility / 3)))
        low = min(open_price, close_price) * (1 - abs(rng.gauss(0, volatility / 3)))
        volume = abs(rng.gauss(1_000, 300)) * (1 + 5 * abs(change))
        candles.append(
            Candle(
                symbol=symbol.upper(),
                interval=interval,
                open_time=open_time,
                close_time=open_time + step - timedelta(milliseconds=1),
                open=round(open_price, 6),
                high=round(high, 6),
                low=round(low, 6),
                close=round(close_price, 6),
                volume=round(volume, 4),
                quote_volume=round(volume * close_price, 2),
                trades=int(volume / 3),
                source="synthetic",
                closed=i < count - 1,
            )
        )
        price = close_price
        open_time += step
    return candles


async def seed_synthetic_market(
    db: AsyncDatabase[dict[str, Any]],
    symbols: list[str],
    *,
    intervals: tuple[str, ...] = ("1m", "1h", "1d"),
    counts: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, int]:
    """Populate candles and asset stats offline (no network). Returns candles written per interval."""
    counts = counts or {"1m": 720, "5m": 576, "15m": 480, "1h": 720, "4h": 360, "1d": 365}
    candles_repo = CandleRepository(db)
    stats_repo = AssetStatsRepository(db)
    written: dict[str, int] = {}
    for index, symbol in enumerate(symbols):
        last_close: float | None = None
        for interval in intervals:
            candles = generate_synthetic_candles(symbol, interval, counts.get(interval, 300), seed=seed + index)
            written[interval] = written.get(interval, 0) + await candles_repo.bulk_upsert(candles)
            last_close = candles[-1].close
            if interval == "1d" and len(candles) > 1:
                day_ago = candles[-2].close
                await stats_repo.upsert_stats(
                    symbol,
                    change_24h_pct=round((candles[-1].close / day_ago - 1) * 100, 3),
                    change_7d_pct=round((candles[-1].close / candles[-8].close - 1) * 100, 3)
                    if len(candles) > 8
                    else None,
                    volume_24h=round(candles[-1].quote_volume, 2),
                    market_cap=round(candles[-1].close * 19_000_000 / (index + 1), 2),
                    rank=index + 1,
                    name=ASSET_NAMES.get(symbol.upper(), symbol.upper()),
                    stats_updated_at=utcnow(),
                )
        if last_close is not None:
            await stats_repo.set_price(symbol, last_close, source="synthetic")
    return written


__all__ = [
    "ASSET_NAMES",
    "REFERENCE_PRICES",
    "ensure_default_assets",
    "generate_synthetic_candles",
    "seed_synthetic_market",
]
