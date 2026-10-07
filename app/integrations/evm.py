"""EVM JSON-RPC client with automatic provider failover (Alchemy -> Infura -> public nodes).

Also contains the small ABI helpers Module 3/6 need: ERC-20 ``Transfer`` log
decoding, ``eth_call`` for token metadata and EIP-55 checksum addresses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import structlog
from Crypto.Hash import keccak

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.metrics import metrics
from app.core.timeutils import utcnow

logger = structlog.get_logger(__name__)

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
SELECTORS = {
    "decimals": "0x313ce567",
    "symbol": "0x95d89b41",
    "name": "0x06fdde03",
    "totalSupply": "0x18160ddd",
    "owner": "0x8da5cb5b",
    "balanceOf": "0x70a08231",
}
ENDPOINT_COOLDOWN_SECONDS = 60.0


def keccak256(data: bytes) -> bytes:
    hasher = keccak.new(digest_bits=256)
    hasher.update(data)
    return hasher.digest()


def to_checksum_address(address: str) -> str:
    addr = address.lower().replace("0x", "")
    digest = keccak256(addr.encode()).hex()
    return "0x" + "".join(c.upper() if int(digest[i], 16) >= 8 else c for i, c in enumerate(addr))


def is_address(value: str | None) -> bool:
    if not value or not isinstance(value, str):
        return False
    value = value.lower()
    return value.startswith("0x") and len(value) == 42 and all(c in "0123456789abcdef" for c in value[2:])


def hex_to_int(value: str | int | None) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return int(value, 16)


def topic_to_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def decode_abi_string(data: str) -> str | None:
    """Decode a single ABI-encoded ``string`` (or a bytes32 symbol) returned by ``eth_call``."""
    raw = data[2:] if data.startswith("0x") else data
    if not raw:
        return None
    try:
        if len(raw) == 64:  # some old tokens return bytes32
            return bytes.fromhex(raw).rstrip(b"\x00").decode("utf-8", errors="ignore") or None
        offset = int(raw[:64], 16) * 2
        length = int(raw[offset : offset + 64], 16) * 2
        return bytes.fromhex(raw[offset + 64 : offset + 64 + length]).decode("utf-8", errors="ignore") or None
    except (ValueError, IndexError):
        return None


@dataclass
class RpcEndpoint:
    name: str
    url: str
    failures: int = 0
    cooldown_until: float = 0.0
    calls: int = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


@dataclass(slots=True)
class TransferLog:
    tx_hash: str
    log_index: int
    block_number: int
    token_address: str
    from_address: str
    to_address: str
    raw_value: int


@dataclass
class ProviderHealth:
    name: str
    url_masked: str
    available: bool
    failures: int
    calls: int
    cooldown_seconds: float
    is_primary: bool
    extra: dict[str, Any] = field(default_factory=dict)


class EvmRpcClient:
    def __init__(self, settings: Settings | None = None, *, timeout: float = 20.0) -> None:
        self.settings = settings or get_settings()
        self.endpoints: list[RpcEndpoint] = []
        if self.settings.alchemy_api_key:
            self.endpoints.append(
                RpcEndpoint(
                    "alchemy",
                    f"https://eth-mainnet.g.alchemy.com/v2/{self.settings.alchemy_api_key.get_secret_value()}",
                )
            )
        if self.settings.infura_api_key:
            self.endpoints.append(
                RpcEndpoint("infura", f"https://mainnet.infura.io/v3/{self.settings.infura_api_key.get_secret_value()}")
            )
        for index, url in enumerate(self.settings.evm_rpc_urls):
            self.endpoints.append(RpcEndpoint(f"custom-{index + 1}", url))
        for url in self.settings.evm_public_rpc_urls:
            self.endpoints.append(RpcEndpoint(httpx.URL(url).host or "public", url))
        self._client = httpx.AsyncClient(timeout=timeout)
        self.stats = metrics.provider("evm_rpc")
        self._request_id = 0
        self.last_endpoint: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def configured(self) -> bool:
        return bool(self.endpoints)

    # ------------------------------------------------------------- core
    async def call(self, method: str, params: list[Any] | None = None) -> Any:
        if not self.endpoints:
            raise ExternalServiceError("No EVM RPC endpoints configured")
        errors: list[str] = []
        ordered = [e for e in self.endpoints if e.available] or self.endpoints
        for endpoint in ordered:
            self._request_id += 1
            payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params or []}
            started = time.perf_counter()
            try:
                response = await self._client.post(endpoint.url, json=payload)
                latency = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                latency = (time.perf_counter() - started) * 1000
                self._mark_failure(endpoint, f"{type(exc).__name__}: {exc}", latency)
                errors.append(f"{endpoint.name}: {type(exc).__name__}")
                continue
            if "error" in body:
                message = str(body["error"].get("message", body["error"]))
                code = body["error"].get("code")
                # Invalid params / execution reverted are the caller's problem, not the node's.
                if code in (-32602, -32000, 3) and "rate" not in message.lower() and "limit" not in message.lower():
                    self.stats.record(latency)
                    endpoint.calls += 1
                    raise ExternalServiceError(f"RPC error: {message}", details={"method": method, "code": code})
                self._mark_failure(endpoint, message, latency)
                errors.append(f"{endpoint.name}: {message[:80]}")
                continue
            endpoint.calls += 1
            endpoint.failures = 0
            self.last_endpoint = endpoint.name
            self.stats.record(latency)
            return body.get("result")
        raise ExternalServiceError("All EVM RPC endpoints failed", details={"errors": errors})

    def _mark_failure(self, endpoint: RpcEndpoint, message: str, latency: float) -> None:
        endpoint.failures += 1
        endpoint.cooldown_until = time.monotonic() + min(ENDPOINT_COOLDOWN_SECONDS * endpoint.failures, 600)
        self.stats.record(latency, error=f"{endpoint.name}: {message}")
        logger.warning("rpc_endpoint_failed", endpoint=endpoint.name, error=message[:200], failures=endpoint.failures)

    def health(self) -> list[ProviderHealth]:
        now = time.monotonic()
        return [
            ProviderHealth(
                name=e.name,
                url_masked=_mask(e.url),
                available=e.available,
                failures=e.failures,
                calls=e.calls,
                cooldown_seconds=round(max(0.0, e.cooldown_until - now), 1),
                is_primary=index == 0,
            )
            for index, e in enumerate(self.endpoints)
        ]

    # --------------------------------------------------------- helpers
    async def block_number(self) -> int:
        return hex_to_int(await self.call("eth_blockNumber"))

    async def get_block(self, number: int | str, *, full_transactions: bool = True) -> dict[str, Any] | None:
        tag = hex(number) if isinstance(number, int) else number
        return await self.call("eth_getBlockByNumber", [tag, full_transactions])  # type: ignore[no-any-return]

    async def get_logs(
        self,
        *,
        from_block: int,
        to_block: int,
        addresses: list[str] | None = None,
        topics: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"fromBlock": hex(from_block), "toBlock": hex(to_block)}
        if addresses:
            params["address"] = addresses if len(addresses) > 1 else addresses[0]
        if topics:
            params["topics"] = topics
        return await self.call("eth_getLogs", [params]) or []  # type: ignore[no-any-return]

    async def get_transfer_logs(
        self, *, from_block: int, to_block: int, tokens: list[str] | None = None
    ) -> list[TransferLog]:
        logs = await self.get_logs(from_block=from_block, to_block=to_block, addresses=tokens, topics=[TRANSFER_TOPIC])
        transfers: list[TransferLog] = []
        for log in logs:
            topics = log.get("topics", [])
            if len(topics) != 3:  # ERC-721 transfers carry a 4th indexed topic; skip them
                continue
            transfers.append(
                TransferLog(
                    tx_hash=log["transactionHash"],
                    log_index=hex_to_int(log.get("logIndex")),
                    block_number=hex_to_int(log.get("blockNumber")),
                    token_address=log["address"].lower(),
                    from_address=topic_to_address(topics[1]),
                    to_address=topic_to_address(topics[2]),
                    raw_value=hex_to_int(log.get("data")) if log.get("data") not in (None, "0x") else 0,
                )
            )
        return transfers

    async def get_code(self, address: str) -> str:
        return await self.call("eth_getCode", [address, "latest"])  # type: ignore[no-any-return]

    async def get_balance(self, address: str) -> int:
        return hex_to_int(await self.call("eth_getBalance", [address, "latest"]))

    async def get_transaction_count(self, address: str) -> int:
        return hex_to_int(await self.call("eth_getTransactionCount", [address, "latest"]))

    async def get_transaction_receipt(self, tx_hash: str) -> dict[str, Any] | None:
        return await self.call("eth_getTransactionReceipt", [tx_hash])  # type: ignore[no-any-return]

    async def eth_call(self, to: str, data: str) -> str:
        return await self.call("eth_call", [{"to": to, "data": data}, "latest"])  # type: ignore[no-any-return]

    async def erc20_metadata(self, token: str) -> dict[str, Any]:
        """Best-effort ``symbol``/``name``/``decimals``/``totalSupply``/``owner`` via ``eth_call``."""
        out: dict[str, Any] = {"address": token.lower()}
        for key in ("symbol", "name"):
            try:
                out[key] = decode_abi_string(await self.eth_call(token, SELECTORS[key]))
            except ExternalServiceError:
                out[key] = None
        try:
            out["decimals"] = hex_to_int(await self.eth_call(token, SELECTORS["decimals"]))
        except ExternalServiceError:
            out["decimals"] = None
        try:
            out["total_supply"] = hex_to_int(await self.eth_call(token, SELECTORS["totalSupply"]))
        except ExternalServiceError:
            out["total_supply"] = None
        try:
            owner_raw = await self.eth_call(token, SELECTORS["owner"])
            out["owner"] = topic_to_address(owner_raw) if owner_raw and len(owner_raw) >= 42 else None
        except ExternalServiceError:
            out["owner"] = None
        return out

    @staticmethod
    def block_time(block: dict[str, Any]) -> datetime:
        ts = hex_to_int(block.get("timestamp"))
        return datetime.fromtimestamp(ts, tz=utcnow().tzinfo)


def _mask(url: str) -> str:
    parsed = httpx.URL(url)
    path = parsed.path
    if len(path) > 12:
        path = path[:8] + "…"
    return f"{parsed.scheme}://{parsed.host}{path}"


__all__ = [
    "SELECTORS",
    "TRANSFER_TOPIC",
    "EvmRpcClient",
    "ProviderHealth",
    "RpcEndpoint",
    "TransferLog",
    "decode_abi_string",
    "hex_to_int",
    "is_address",
    "keccak256",
    "to_checksum_address",
    "topic_to_address",
]
