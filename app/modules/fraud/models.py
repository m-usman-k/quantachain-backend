"""Module 6 persistence models: contract scans, fraud alerts, watched pairs, project scores, blacklist."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.db.models import Document


class AlertType(StrEnum):
    RUG_PULL = "rug_pull"
    PUMP_AND_DUMP = "pump_and_dump"
    CONTRACT_VULNERABILITY = "contract_vulnerability"
    BLACKLISTED_WALLET = "blacklisted_wallet"
    SUSPICIOUS_LAUNCH = "suspicious_launch"
    LIQUIDITY_CYCLE = "liquidity_cycle"


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FindingSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RiskLabel(StrEnum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ScoreLabel(StrEnum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DANGEROUS = "dangerous"


class BlacklistStatus(StrEnum):
    BLOCKED = "blocked"
    MONITORING = "monitoring"
    CLEARED = "cleared"


class BlacklistSource(StrEnum):
    MANUAL = "manual"
    MITIGATION = "mitigation"
    IMPORT = "import"


class Finding(BaseModel):
    """One scanner detection. ``swc`` is set for SWC-registry weaknesses, ``None`` for custom indicators."""

    id: str
    swc: str | None = None
    title: str
    severity: FindingSeverity
    category: str = "vulnerability"  # vulnerability | centralisation | honeypot | info
    description: str
    recommendation: str
    lines: list[int] = Field(default_factory=list)
    snippet: str | None = None
    function: str | None = None
    contract: str | None = None


class ContractScan(Document):
    user_id: str
    chain: str = "eth"
    address: str | None = None
    contract_name: str | None = None
    verified: bool = False
    compiler_version: str | None = None
    is_proxy: bool = False
    implementation: str | None = None
    source_hash: str | None = None
    kind: str = "source"  # source | bytecode
    findings: list[Finding] = Field(default_factory=list)
    risk_score: int = 0
    risk_label: RiskLabel = RiskLabel.SAFE
    stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)  # erc20 metadata, source origin
    liquidity_context: dict[str, Any] = Field(default_factory=dict)  # best DEX pair snapshot
    project_score_id: str | None = None
    duration_ms: float = 0.0


class FraudAlert(Document):
    """Cross-module contract: see docs/ARCHITECTURE.md (Fraud alerts)."""

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
    mitigation: dict[str, Any] | None = None  # {by, at, note, blacklisted: [...]}


class WatchedPair(Document):
    chain: str  # DexScreener chain id (ethereum | bsc | solana | ...)
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


class ScoreComponent(BaseModel):
    name: str
    score: float
    max: float
    detail: str


class ProjectScore(Document):
    chain: str
    address: str
    symbol: str | None = None
    score: int
    label: ScoreLabel
    components: list[ScoreComponent] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime


class BlacklistEntry(Document):
    """AML wallet blacklist row (read directly by Module 3 - keep field names stable)."""

    address: str  # lowercase for EVM
    chain: str = "eth"
    risk_score: int = Field(default=80, ge=0, le=100)
    reason: str
    status: BlacklistStatus = BlacklistStatus.BLOCKED
    source: BlacklistSource = BlacklistSource.MANUAL
    added_by: str | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


__all__ = [
    "AlertSeverity",
    "AlertType",
    "BlacklistEntry",
    "BlacklistSource",
    "BlacklistStatus",
    "ContractScan",
    "Finding",
    "FindingSeverity",
    "FraudAlert",
    "ProjectScore",
    "RiskLabel",
    "ScoreComponent",
    "ScoreLabel",
    "WatchedPair",
]
