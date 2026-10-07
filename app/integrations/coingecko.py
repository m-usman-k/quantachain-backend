"""CoinGecko market data and the alternative.me Fear & Greed index."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.integrations.base import ProviderClient

# Well-known CoinGecko ids for the default tracked universe.
DEFAULT_COINGECKO_IDS: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "POL": "polygon-ecosystem-token",
    "TRX": "tron",
    "LTC": "litecoin",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "NEAR": "near",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "APT": "aptos",
    "PEPE": "pepe",
    "SHIB": "shiba-inu",
    "USDT": "tether",
    "USDC": "usd-coin",
    "WBTC": "wrapped-bitcoin",
}


class CoinGeckoClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        headers = {}
        if settings.coingecko_api_key:
            headers["x-cg-demo-api-key"] = settings.coingecko_api_key.get_secret_value()
        super().__init__("coingecko", settings.coingecko_api_url, headers=headers, timeout=20.0)

    async def coins_markets(self, ids: list[str], *, vs_currency: str = "usd") -> list[dict[str, Any]]:
        if not ids:
            return []
        params = {
            "vs_currency": vs_currency,
            "ids": ",".join(ids),
            "order": "market_cap_desc",
            "per_page": min(len(ids), 250),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        return await self.get_json("/coins/markets", params=params)  # type: ignore[no-any-return]

    async def top_markets(self, *, per_page: int = 50, vs_currency: str = "usd") -> list[dict[str, Any]]:
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": min(per_page, 250),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        return await self.get_json("/coins/markets", params=params)  # type: ignore[no-any-return]

    async def global_data(self) -> dict[str, Any]:
        payload = await self.get_json("/global")
        return payload.get("data", payload)  # type: ignore[no-any-return]

    async def market_chart(self, coin_id: str, *, days: int = 30, vs_currency: str = "usd") -> dict[str, Any]:
        return await self.get_json(  # type: ignore[no-any-return]
            f"/coins/{coin_id}/market_chart", params={"vs_currency": vs_currency, "days": days}
        )

    async def search(self, query: str) -> list[dict[str, Any]]:
        payload = await self.get_json("/search", params={"query": query})
        return payload.get("coins", [])  # type: ignore[no-any-return]

    async def coin_info(self, coin_id: str) -> dict[str, Any]:
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
            "sparkline": "false",
        }
        return await self.get_json(f"/coins/{coin_id}", params=params)  # type: ignore[no-any-return]


class FearGreedClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__("fear_greed", settings.fear_greed_api_url, timeout=10.0)

    async def latest(self, limit: int = 1) -> list[dict[str, Any]]:
        payload = await self.get_json("", params={"limit": limit, "format": "json"})
        rows = payload.get("data", [])
        return [
            {
                "value": int(row["value"]),
                "label": row.get("value_classification"),
                "timestamp": int(row["timestamp"]),
            }
            for row in rows
        ]


__all__ = ["DEFAULT_COINGECKO_IDS", "CoinGeckoClient", "FearGreedClient"]
