"""Moralis Web3 Data API (REST) and Streams webhook verification."""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings, get_settings
from app.integrations.base import ProviderClient
from app.integrations.evm import keccak256

MORALIS_BASE = "https://deep-index.moralis.io/api/v2.2"


class MoralisClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        headers = {}
        if settings.moralis_api_key:
            headers["X-API-Key"] = settings.moralis_api_key.get_secret_value()
        super().__init__("moralis", MORALIS_BASE, headers=headers, timeout=20.0)
        self.configured = settings.moralis_api_key is not None
        self.chain = settings.onchain_chain if hasattr(settings, "onchain_chain") else "eth"
        self._stream_secret = (
            settings.moralis_stream_secret.get_secret_value() if settings.moralis_stream_secret else None
        )

    async def wallet_token_transfers(
        self, address: str, *, limit: int = 100, cursor: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"chain": self.chain, "limit": min(limit, 100), "order": "DESC"}
        if cursor:
            params["cursor"] = cursor
        return await self.get_json(f"/{address}/erc20/transfers", params=params)  # type: ignore[no-any-return]

    async def wallet_transactions(self, address: str, *, limit: int = 100, cursor: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"chain": self.chain, "limit": min(limit, 100), "order": "DESC"}
        if cursor:
            params["cursor"] = cursor
        return await self.get_json(f"/{address}", params=params)  # type: ignore[no-any-return]

    async def wallet_stats(self, address: str) -> dict[str, Any]:
        return await self.get_json(f"/wallets/{address}/stats", params={"chain": self.chain})  # type: ignore[no-any-return]

    async def wallet_net_worth(self, address: str) -> dict[str, Any]:
        return await self.get_json(
            f"/wallets/{address}/net-worth", params={"chain": self.chain, "exclude_spam": "true"}
        )  # type: ignore[no-any-return]

    async def token_price(self, token_address: str) -> dict[str, Any]:
        return await self.get_json(f"/erc20/{token_address}/price", params={"chain": self.chain})  # type: ignore[no-any-return]

    async def token_metadata(self, addresses: list[str]) -> list[dict[str, Any]]:
        params = [("chain", self.chain)] + [("addresses[]", a) for a in addresses]
        return await self.get_json("/erc20/metadata", params=params)  # type: ignore[no-any-return]

    async def token_owners(self, token_address: str, *, limit: int = 100, cursor: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"chain": self.chain, "limit": min(limit, 100), "order": "DESC"}
        if cursor:
            params["cursor"] = cursor
        return await self.get_json(f"/erc20/{token_address}/owners", params=params)  # type: ignore[no-any-return]

    async def token_transfers(
        self, token_address: str, *, limit: int = 100, cursor: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"chain": self.chain, "limit": min(limit, 100), "order": "DESC"}
        if cursor:
            params["cursor"] = cursor
        return await self.get_json(f"/erc20/{token_address}/transfers", params=params)  # type: ignore[no-any-return]

    async def token_stats(self, token_address: str) -> dict[str, Any]:
        return await self.get_json(f"/erc20/{token_address}/stats", params={"chain": self.chain})  # type: ignore[no-any-return]

    # ------------------------------------------------------------ streams
    def verify_stream_signature(self, raw_body: bytes, signature: str | None) -> bool:
        """Moralis signs webhooks with keccak256(body + secret) in the ``x-signature`` header."""
        if not self._stream_secret or not signature:
            return False
        expected = "0x" + keccak256(raw_body + self._stream_secret.encode()).hex()
        return expected.lower() == signature.lower()

    @staticmethod
    def parse_stream_payload(body: bytes) -> dict[str, Any]:
        return json.loads(body or b"{}")  # type: ignore[no-any-return]


__all__ = ["MORALIS_BASE", "MoralisClient"]
