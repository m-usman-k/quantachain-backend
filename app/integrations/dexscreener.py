"""DexScreener public API - DEX pair liquidity, volume and price changes (no key required)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.config import Settings, get_settings
from app.core.timeutils import from_ms
from app.integrations.base import ProviderClient


@dataclass
class DexPair:
    chain: str
    dex: str
    pair_address: str
    url: str | None
    base_symbol: str | None
    base_name: str | None
    base_address: str | None
    quote_symbol: str | None
    quote_address: str | None
    price_usd: float | None
    price_native: float | None
    liquidity_usd: float | None
    liquidity_base: float | None
    liquidity_quote: float | None
    fdv: float | None
    market_cap: float | None
    volume: dict[str, float] = field(default_factory=dict)  # m5/h1/h6/h24
    price_change: dict[str, float] = field(default_factory=dict)  # m5/h1/h6/h24 (%)
    txns: dict[str, dict[str, int]] = field(default_factory=dict)  # window -> {buys, sells}
    created_at: datetime | None = None

    @property
    def age_hours(self) -> float | None:
        if self.created_at is None:
            return None
        from app.core.timeutils import utcnow

        return max(0.0, (utcnow() - self.created_at).total_seconds() / 3600)

    def buy_sell_ratio(self, window: str = "h1") -> float | None:
        tx = self.txns.get(window)
        if not tx:
            return None
        sells = tx.get("sells", 0)
        buys = tx.get("buys", 0)
        if buys + sells == 0:
            return None
        return buys / max(sells, 1)


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_pair(raw: dict[str, Any]) -> DexPair:
    base = raw.get("baseToken") or {}
    quote = raw.get("quoteToken") or {}
    liquidity = raw.get("liquidity") or {}
    created = raw.get("pairCreatedAt")
    return DexPair(
        chain=raw.get("chainId", ""),
        dex=raw.get("dexId", ""),
        pair_address=(raw.get("pairAddress") or "").lower(),
        url=raw.get("url"),
        base_symbol=base.get("symbol"),
        base_name=base.get("name"),
        base_address=(base.get("address") or "").lower() or None,
        quote_symbol=quote.get("symbol"),
        quote_address=(quote.get("address") or "").lower() or None,
        price_usd=_f(raw.get("priceUsd")),
        price_native=_f(raw.get("priceNative")),
        liquidity_usd=_f(liquidity.get("usd")),
        liquidity_base=_f(liquidity.get("base")),
        liquidity_quote=_f(liquidity.get("quote")),
        fdv=_f(raw.get("fdv")),
        market_cap=_f(raw.get("marketCap")),
        volume={k: v for k, v in ((k, _f(v)) for k, v in (raw.get("volume") or {}).items()) if v is not None},
        price_change={
            k: v for k, v in ((k, _f(v)) for k, v in (raw.get("priceChange") or {}).items()) if v is not None
        },
        txns={
            k: {"buys": int(v.get("buys", 0)), "sells": int(v.get("sells", 0))}
            for k, v in (raw.get("txns") or {}).items()
            if isinstance(v, dict)
        },
        created_at=from_ms(int(created)) if created else None,
    )


class DexScreenerClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__("dexscreener", settings.dexscreener_api_url, timeout=20.0)

    async def token_pairs(self, token_addresses: list[str]) -> list[DexPair]:
        if not token_addresses:
            return []
        payload = await self.get_json(f"/latest/dex/tokens/{','.join(token_addresses[:30])}")
        return [parse_pair(p) for p in payload.get("pairs") or []]

    async def pair(self, chain: str, pair_address: str) -> DexPair | None:
        payload = await self.get_json(f"/latest/dex/pairs/{chain}/{pair_address}")
        pairs = payload.get("pairs") or ([payload["pair"]] if payload.get("pair") else [])
        return parse_pair(pairs[0]) if pairs else None

    async def search(self, query: str) -> list[DexPair]:
        payload = await self.get_json("/latest/dex/search", params={"q": query})
        return [parse_pair(p) for p in payload.get("pairs") or []]

    async def latest_token_profiles(self) -> list[dict[str, Any]]:
        payload = await self.get_json("/token-profiles/latest/v1")
        return payload if isinstance(payload, list) else []

    async def top_boosted_tokens(self) -> list[dict[str, Any]]:
        payload = await self.get_json("/token-boosts/top/v1")
        return payload if isinstance(payload, list) else []


__all__ = ["DexPair", "DexScreenerClient", "parse_pair"]
