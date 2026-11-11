"""Module 3 persistence models: whale transfers, wallet profiles, cluster graphs, and metrics."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.db.models import Document


class TransferFlow(StrEnum):
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    EXCHANGE_INTERNAL = "exchange_internal"
    WALLET_TO_WALLET = "wallet_to_wallet"


class TransferRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    SUSPICIOUS = "suspicious"


class ConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REPLACED = "replaced"


class EntityType(StrEnum):
    EXCHANGE = "exchange"
    WHALE = "whale"
    SMART_MONEY = "smart_money"
    FUND = "fund"
    CONTRACT = "contract"
    INDIVIDUAL = "individual"
    UNKNOWN = "unknown"


class WhaleTransfer(Document):
    """One high-value or smart-money on-chain transfer."""

    tx_hash: str
    log_index: int = 0
    chain: str = "eth"
    block_number: int
    block_time: datetime
    symbol: str
    token_address: str | None = None
    amount: float
    amount_usd: float | None = None
    price_source: str | None = None
    price_timestamp: datetime | None = None
    from_address: str
    to_address: str
    from_label: str | None = None
    to_label: str | None = None
    flow: TransferFlow = TransferFlow.WALLET_TO_WALLET
    risk_level: TransferRisk = TransferRisk.LOW
    confirmation_status: ConfirmationStatus = ConfirmationStatus.CONFIRMED
    gas_fee_eth: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Wallet(Document):
    """Tracked wallet or smart money entity profile."""

    address: str
    chain: str = "eth"
    label: str | None = None
    entity_type: EntityType = EntityType.UNKNOWN
    balance_usd: float | None = None
    smart_money_score: int = Field(default=50, ge=0, le=100)
    win_rate: float | None = None
    total_profit_usd: float | None = None
    total_volume_usd: float | None = None
    transaction_count: int = 0
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    is_tracked: bool = True
    is_exchange: bool = False


class GraphNode(BaseModel):
    id: str
    label: str | None = None
    entity_type: EntityType = EntityType.UNKNOWN
    balance_usd: float | None = None
    smart_money_score: int | None = None
    risk_level: TransferRisk = TransferRisk.LOW


class GraphEdge(BaseModel):
    source: str
    target: str
    amount: float
    amount_usd: float | None = None
    symbol: str
    tx_hash: str
    block_time: datetime


class WalletClusterGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    center_address: str | None = None
    total_volume_usd: float = 0.0


__all__ = [
    "ConfirmationStatus",
    "EntityType",
    "GraphEdge",
    "GraphNode",
    "TransferFlow",
    "TransferRisk",
    "Wallet",
    "WalletClusterGraph",
    "WhaleTransfer",
]
