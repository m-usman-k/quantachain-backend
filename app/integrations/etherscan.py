"""Etherscan API V2 (verified source code, ABI, contract creation, transactions)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.integrations.base import ProviderClient


@dataclass
class VerifiedSource:
    address: str
    contract_name: str | None
    compiler_version: str | None
    license: str | None
    optimization_used: bool
    is_proxy: bool
    implementation: str | None
    source_files: dict[str, str] = field(default_factory=dict)  # path -> solidity source
    abi: list[dict[str, Any]] | None = None

    @property
    def verified(self) -> bool:
        return bool(self.source_files)

    @property
    def combined_source(self) -> str:
        return "\n\n".join(f"// File: {path}\n{code}" for path, code in self.source_files.items())


class EtherscanClient(ProviderClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        params = {"chainid": str(settings.evm_chain_id)}
        if settings.etherscan_api_key:
            params["apikey"] = settings.etherscan_api_key.get_secret_value()
        super().__init__("etherscan", settings.etherscan_api_url, params=params, timeout=20.0)
        self.configured = settings.etherscan_api_key is not None

    async def _query(self, **params: Any) -> Any:
        payload = await self.get_json("", params=params)
        if str(payload.get("status")) != "1" and payload.get("message") not in ("OK", "No transactions found"):
            result = payload.get("result")
            if isinstance(result, str) and "rate limit" in result.lower():
                raise ExternalServiceError("Etherscan rate limit reached")
            if payload.get("message") == "NOTOK" and isinstance(result, str):
                raise ExternalServiceError(f"Etherscan: {result}")
        return payload.get("result")

    async def get_source(self, address: str) -> VerifiedSource:
        result = await self._query(module="contract", action="getsourcecode", address=address)
        row = (result or [{}])[0] if isinstance(result, list) else {}
        sources = _parse_sources(row.get("SourceCode") or "", row.get("ContractName") or "Contract")
        abi: list[dict[str, Any]] | None = None
        raw_abi = row.get("ABI")
        if raw_abi and raw_abi.startswith("["):
            try:
                abi = json.loads(raw_abi)
            except ValueError:
                abi = None
        return VerifiedSource(
            address=address.lower(),
            contract_name=row.get("ContractName") or None,
            compiler_version=row.get("CompilerVersion") or None,
            license=row.get("LicenseType") or None,
            optimization_used=str(row.get("OptimizationUsed")) == "1",
            is_proxy=str(row.get("Proxy")) == "1",
            implementation=(row.get("Implementation") or None),
            source_files=sources,
            abi=abi,
        )

    async def get_contract_creation(self, addresses: list[str]) -> list[dict[str, Any]]:
        result = await self._query(
            module="contract", action="getcontractcreation", contractaddresses=",".join(addresses)
        )
        return result if isinstance(result, list) else []

    async def get_transactions(self, address: str, *, limit: int = 100, sort: str = "desc") -> list[dict[str, Any]]:
        result = await self._query(
            module="account", action="txlist", address=address, page=1, offset=min(limit, 10_000), sort=sort
        )
        return result if isinstance(result, list) else []

    async def get_token_transfers(
        self, address: str, *, contract: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "page": 1,
            "offset": min(limit, 10_000),
            "sort": "desc",
        }
        if contract:
            params["contractaddress"] = contract
        result = await self._query(**params)
        return result if isinstance(result, list) else []


def _parse_sources(raw: str, contract_name: str) -> dict[str, str]:
    """Etherscan returns either flat Solidity, a JSON map, or standard-JSON input wrapped in ``{{ }}``."""
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return {f"{contract_name}.sol": raw}
        sources = data.get("sources", data) if isinstance(data, dict) else {}
        out: dict[str, str] = {}
        for path, entry in sources.items():
            content = entry.get("content") if isinstance(entry, dict) else entry
            if isinstance(content, str):
                out[path] = content
        return out or {f"{contract_name}.sol": raw}
    return {f"{contract_name}.sol": text}


__all__ = ["EtherscanClient", "VerifiedSource"]
