"""Module 6 request/response schemas for contract scanning, fraud alerts, watched pairs, and blacklist."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.fraud.models import (
    AlertSeverity,
    AlertType,
    BlacklistSource,
    BlacklistStatus,
    Finding,
    RiskLabel,
    ScoreComponent,
    ScoreLabel,
)


class ScanIn(BaseModel):
    chain: str = "eth"
    address: str | None = None
    source_code: str | None = None
    contract_name: str | None = None
    compiler_version: str | None = None


class ScanOut(BaseModel):
    id: str
    user_id: str
    chain: str
    address: str | None = None
    contract_name: str | None = None
    verified: bool = False
    compiler_version: str | None = None
    is_proxy: bool = False
    implementation: str | None = None
    source_hash: str | None = None
    kind: str = "source"
    findings: list[Finding] = Field(default_factory=list)
    risk_score: int
    risk_label: RiskLabel
    stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    liquidity_context: dict[str, Any] = Field(default_factory=dict)
    project_score_id: str | None = None
    duration_ms: float = 0.0
    created_at: datetime


class FraudAlertOut(BaseModel):
    id: str
    type: AlertType
    severity: AlertSeverity
    token_address: str | None = None
    pair_address: str | None = None
    chain: str = "ethereum"
    symbol: str | None = None
    title: str
    body: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str
    risk_score: int = 0
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    mitigation: dict[str, Any] | None = None
    created_at: datetime


class FraudAlertAck(BaseModel):
    note: str | None = None


class WatchedPairIn(BaseModel):
    chain: str
    pair_address: str
    token_address: str | None = None
    base_symbol: str | None = None
    quote_symbol: str | None = None
    dex: str | None = None
    label: str | None = None


class WatchedPairOut(BaseModel):
    id: str
    chain: str
    pair_address: str
    token_address: str | None = None
    base_symbol: str | None = None
    quote_symbol: str | None = None
    dex: str | None = None
    label: str | None = None
    added_by: str | None = None
    auto_added: bool = False
    active: bool = True
    last_checked_at: datetime | None = None
    last_liquidity_usd: float | None = None
    last_price_usd: float | None = None
    pair_created_at: datetime | None = None
    created_at: datetime


class BlacklistIn(BaseModel):
    address: str
    chain: str = "eth"
    risk_score: int = Field(default=80, ge=0, le=100)
    reason: str
    status: BlacklistStatus = BlacklistStatus.BLOCKED
    source: BlacklistSource = BlacklistSource.MANUAL
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


class BlacklistOut(BaseModel):
    id: str
    address: str
    chain: str
    risk_score: int
    reason: str
    status: BlacklistStatus
    source: BlacklistSource
    added_by: str | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None


class ProjectScoreOut(BaseModel):
    id: str
    chain: str
    address: str
    symbol: str | None = None
    score: int
    label: ScoreLabel
    components: list[ScoreComponent] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime


class FraudStatsOut(BaseModel):
    total_scans: int
    recent_scans_24h: int
    alerts_by_severity: dict[str, int]
    monitored_pairs: int
    blacklisted_addresses: int
    average_project_score: float | None = None


__all__ = [
    "BlacklistIn",
    "BlacklistOut",
    "FraudAlertAck",
    "FraudAlertOut",
    "FraudStatsOut",
    "ProjectScoreOut",
    "ScanIn",
    "ScanOut",
    "WatchedPairIn",
    "WatchedPairOut",
]
