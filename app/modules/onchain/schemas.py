"""Module 3 API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.onchain.models import (
    ConfirmationStatus,
    EntityType,
    GraphEdge,
    GraphNode,
    TransferFlow,
    TransferRisk,
)


class WhaleTransferOut(BaseModel):
    id: str
    tx_hash: str
    log_index: int = 0
    chain: str
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
    flow: TransferFlow
    risk_level: TransferRisk
    confirmation_status: ConfirmationStatus
    gas_fee_eth: float | None = None
    created_at: datetime


class WalletProfileOut(BaseModel):
    id: str
    address: str
    chain: str
    label: str | None = None
    entity_type: EntityType
    balance_usd: float | None = None
    smart_money_score: int
    win_rate: float | None = None
    total_profit_usd: float | None = None
    total_volume_usd: float | None = None
    transaction_count: int
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    is_tracked: bool
    is_exchange: bool
    recent_transfers: list[WhaleTransferOut] = Field(default_factory=list)


class ClusterGraphOut(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    center_address: str | None = None
    total_volume_usd: float = 0.0


class NetflowOut(BaseModel):
    symbol: str
    window_hours: int
    inflow_usd: float
    outflow_usd: float
    netflow_usd: float
    trend: str  # accumulating | distributing | neutral
    updated_at: datetime


class InvestigationIn(BaseModel):
    tx_hash: str | None = None
    address: str | None = None
    chain: str = "eth"


class InvestigationOut(BaseModel):
    query: str
    entity_type: str  # transaction | wallet
    details: dict[str, Any]
    graph: ClusterGraphOut
    risk_assessment: dict[str, Any]
    provenance: dict[str, Any]


__all__ = [
    "ClusterGraphOut",
    "InvestigationIn",
    "InvestigationOut",
    "NetflowOut",
    "WalletProfileOut",
    "WhaleTransferOut",
]
