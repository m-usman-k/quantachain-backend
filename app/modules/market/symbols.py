"""Detects which tracked assets a piece of text talks about."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.modules.market.models import Asset

# Names that are ordinary English words and must not be matched case-insensitively.
AMBIGUOUS_NAMES = {"near", "sui", "op", "apt", "pol", "uni", "bnb", "ton", "core", "flow", "gas", "blur", "stars"}

ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "ether"],
    "SOL": ["solana"],
    "XRP": ["ripple", "xrp"],
    "DOGE": ["dogecoin"],
    "ADA": ["cardano"],
    "AVAX": ["avalanche"],
    "LINK": ["chainlink"],
    "DOT": ["polkadot"],
    "MATIC": ["polygon"],
    "POL": ["polygon"],
    "BNB": ["binance coin"],
    "USDT": ["tether"],
    "USDC": ["usd coin"],
    "LTC": ["litecoin"],
    "TRX": ["tron"],
    "ATOM": ["cosmos"],
    "ARB": ["arbitrum"],
    "OP": ["optimism"],
    "SHIB": ["shiba inu"],
}


class SymbolDetector:
    def __init__(self, assets: Iterable[Asset]) -> None:
        self._symbol_patterns: list[tuple[str, re.Pattern[str]]] = []
        self._name_patterns: list[tuple[str, re.Pattern[str]]] = []
        for asset in assets:
            symbol = asset.symbol.upper()
            self._symbol_patterns.append(
                (symbol, re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(symbol)}(?![A-Za-z0-9])"))
            )
            names = {asset.name.lower()} | set(ALIASES.get(symbol, []))
            for name in names:
                if len(name) < 3 or name in AMBIGUOUS_NAMES:
                    continue
                self._name_patterns.append((symbol, re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)))

    def detect(self, *texts: str | None) -> list[str]:
        text = " ".join(t for t in texts if t)
        if not text:
            return []
        found: dict[str, None] = {}
        for symbol, pattern in self._symbol_patterns:
            if pattern.search(text):
                found[symbol] = None
        for symbol, pattern in self._name_patterns:
            if symbol not in found and pattern.search(text):
                found[symbol] = None
        return list(found)


__all__ = ["ALIASES", "AMBIGUOUS_NAMES", "SymbolDetector"]
