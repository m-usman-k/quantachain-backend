"""Module 6 business logic: contract scanning, fraud alerts, project score, DEX monitoring, AML blacklist."""

from __future__ import annotations

import hashlib
import time
from datetime import timedelta
from typing import Any

import structlog

from app.core.config import Settings, get_settings
from app.core.events import Topics, event_bus
from app.core.exceptions import ValidationFailedError
from app.core.timeutils import utcnow
from app.integrations.dexscreener import DexPair, DexScreenerClient
from app.integrations.etherscan import EtherscanClient
from app.integrations.evm import EvmRpcClient, is_address
from app.modules.fraud.models import (
    AlertSeverity,
    AlertType,
    BlacklistEntry,
    BlacklistStatus,
    ContractScan,
    Finding,
    FindingSeverity,
    FraudAlert,
    ProjectScore,
    RiskLabel,
    ScoreComponent,
    ScoreLabel,
    WatchedPair,
)
from app.modules.fraud.repository import (
    BlacklistRepository,
    ContractScanRepository,
    FraudAlertRepository,
    LiquiditySnapshotRepository,
    ProjectScoreRepository,
    WatchedPairRepository,
)
from app.modules.fraud.scanner import BytecodeScanner, SolidityScanner
from app.modules.fraud.schemas import BlacklistIn, ScanIn, WatchedPairIn
from app.modules.notifications.models import NotificationType, Severity
from app.modules.notifications.service import NotificationService

logger = structlog.get_logger(__name__)


class FraudService:
    def __init__(self, db: Any, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.scans = ContractScanRepository(db)
        self.alerts = FraudAlertRepository(db)
        self.pairs = WatchedPairRepository(db)
        self.snapshots = LiquiditySnapshotRepository(db)
        self.scores = ProjectScoreRepository(db)
        self.blacklist = BlacklistRepository(db)
        self.etherscan = EtherscanClient(self.settings)
        self.dexscreener = DexScreenerClient(self.settings)
        self.evm = EvmRpcClient(self.settings)
        self.notifications = NotificationService(db, self.settings)

    # ----------------------------------------------------------- smart contract scan
    async def scan_contract(self, user_id: str, payload: ScanIn) -> ContractScan:
        started = time.perf_counter()
        source = payload.source_code
        address = payload.address.lower() if payload.address else None
        chain = payload.chain.lower()
        verified = False
        compiler_version = payload.compiler_version
        contract_name = payload.contract_name
        is_proxy = False
        implementation: str | None = None
        kind = "source"
        findings: list[Finding] = []
        stats: dict[str, Any] = {}
        liquidity_context: dict[str, Any] = {}

        if not source and not address:
            raise ValidationFailedError("Either source_code or contract address must be provided")

        if address and not is_address(address):
            raise ValidationFailedError(f"Invalid contract address: {address}")

        # Check blacklist first
        if address:
            bl_entry = await self.blacklist.get_by_address(address)
            if bl_entry and bl_entry.status == BlacklistStatus.BLOCKED:
                findings.append(
                    Finding(
                        id="blacklisted-address",
                        title="Address On AML Blacklist",
                        severity=FindingSeverity.CRITICAL,
                        category="blacklist",
                        description=f"Address {address} is flagged on the internal AML blacklist: {bl_entry.reason}",
                        recommendation="Do not interact with this contract or transfer assets.",
                    )
                )

        # Retrieve source from Etherscan if only address is provided
        if not source and address:
            try:
                verified_data = await self.etherscan.get_source(address)
                if verified_data.verified:
                    source = verified_data.combined_source
                    verified = True
                    contract_name = verified_data.contract_name or contract_name
                    compiler_version = verified_data.compiler_version or compiler_version
                    is_proxy = verified_data.is_proxy
                    implementation = verified_data.implementation
            except Exception as exc:
                logger.warning("etherscan_source_failed", address=address, error=str(exc))

        # Also get liquidity context from DexScreener if address is present
        if address:
            try:
                dex_pairs = await self.dexscreener.token_pairs(chain, address)
                if dex_pairs:
                    top_pair = max(dex_pairs, key=lambda p: p.liquidity_usd or 0.0)
                    liquidity_context = {
                        "pair_address": top_pair.pair_address,
                        "dex": top_pair.dex,
                        "liquidity_usd": top_pair.liquidity_usd,
                        "price_usd": top_pair.price_usd,
                        "base_symbol": top_pair.base_symbol,
                        "quote_symbol": top_pair.quote_symbol,
                        "created_at": top_pair.created_at.isoformat() if top_pair.created_at else None,
                        "volume_h24": top_pair.volume.get("h24"),
                    }
            except Exception as exc:
                logger.warning("dexscreener_context_failed", address=address, error=str(exc))

        # Run scanner
        if source:
            scanner = SolidityScanner()
            result = scanner.scan(source)
            findings.extend(result.findings)
            risk_score = result.risk_score
            risk_label = result.risk_label
            stats = result.stats
            source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            kind = "source"
        elif address:
            # Unverified contract: fallback to BytecodeScanner
            kind = "bytecode"
            source_hash = None
            try:
                bytecode = await self.evm.get_code(address)
                if bytecode and bytecode != "0x":
                    b_scanner = BytecodeScanner()
                    result = b_scanner.scan(bytecode)
                    findings.extend(result.findings)
                    risk_score = result.risk_score
                    risk_label = result.risk_label
                    stats = result.stats
                else:
                    risk_score = 0
                    risk_label = RiskLabel.SAFE
                    findings.append(
                        Finding(
                            id="empty-bytecode",
                            title="Empty Bytecode or Not a Contract",
                            severity=FindingSeverity.INFO,
                            category="info",
                            description="The target address contains no EVM bytecode.",
                            recommendation="Ensure target is a deployed smart contract.",
                        )
                    )
            except Exception as exc:
                logger.warning("bytecode_fetch_failed", address=address, error=str(exc))
                risk_score = 50
                risk_label = RiskLabel.MEDIUM
                findings.append(
                    Finding(
                        id="unverified-unavailable",
                        title="Unverified Contract / RPC Error",
                        severity=FindingSeverity.MEDIUM,
                        category="info",
                        description=f"Source code unverified on Etherscan and RPC query failed: {exc}",
                        recommendation="Exercise caution before interacting with unverified contracts.",
                    )
                )
        else:
            raise ValidationFailedError("Unable to analyze contract: no source or bytecode found")

        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        # Project score calculation if address is available
        project_score_id: str | None = None
        if address:
            try:
                p_score = await self.compute_project_score(chain, address)
                project_score_id = p_score.id
            except Exception as exc:
                logger.warning("project_score_failed", address=address, error=str(exc))

        scan = ContractScan(
            user_id=user_id,
            chain=chain,
            address=address,
            contract_name=contract_name,
            verified=verified,
            compiler_version=compiler_version,
            is_proxy=is_proxy,
            implementation=implementation,
            source_hash=source_hash,
            kind=kind,
            findings=findings,
            risk_score=risk_score,
            risk_label=risk_label,
            stats=stats,
            metadata={"source_length": len(source) if source else 0},
            liquidity_context=liquidity_context,
            project_score_id=project_score_id,
            duration_ms=duration_ms,
        )
        saved = await self.scans.insert(scan)

        # Trigger alert if critical vulnerability detected
        critical_findings = [f for f in findings if f.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH)]
        if critical_findings and address:
            fp = f"contract_vuln:{chain}:{address}:{risk_score}"
            alert = FraudAlert(
                type=AlertType.CONTRACT_VULNERABILITY,
                severity=AlertSeverity.CRITICAL if risk_score >= 70 else AlertSeverity.WARNING,
                token_address=address,
                chain=chain,
                symbol=contract_name,
                title=f"Vulnerability Alert: {contract_name or address[:10]}",
                body=f"Contract scan detected {len(critical_findings)} high/critical security issues with risk score {risk_score}.",
                evidence={
                    "findings_count": len(findings),
                    "critical_count": len(critical_findings),
                    "top_findings": [f.title for f in critical_findings[:3]],
                    "scan_id": saved.id,
                },
                fingerprint=fp,
                risk_score=risk_score,
            )
            await self.create_alert(alert)

        return saved

    async def get_scan(self, scan_id: str) -> ContractScan:
        return await self.scans.get_or_raise(scan_id)

    async def latest_scan_for_address(self, chain: str, address: str) -> ContractScan | None:
        return await self.scans.latest_for_address(chain.lower(), address.lower())

    # ----------------------------------------------------------- project score
    async def compute_project_score(self, chain: str, address: str, symbol: str | None = None) -> ProjectScore:
        chain = chain.lower()
        address = address.lower()
        components: list[ScoreComponent] = []

        # 1. DEX Liquidity Depth (25 max)
        pairs: list[DexPair] = []
        try:
            pairs = await self.dexscreener.token_pairs(chain, address)
        except Exception:
            pass

        best_pair = max(pairs, key=lambda p: p.liquidity_usd or 0.0) if pairs else None
        liq_usd = best_pair.liquidity_usd if best_pair else 0.0
        if liq_usd >= 100_000:
            liq_score = 25.0
            liq_detail = f"High liquidity: ${liq_usd:,.0f}"
        elif liq_usd >= 30_000:
            liq_score = 18.0
            liq_detail = f"Moderate liquidity: ${liq_usd:,.0f}"
        elif liq_usd >= 10_000:
            liq_score = 10.0
            liq_detail = f"Low liquidity: ${liq_usd:,.0f}"
        elif liq_usd >= 1_000:
            liq_score = 5.0
            liq_detail = f"Very thin liquidity: ${liq_usd:,.0f}"
        else:
            liq_score = 0.0
            liq_detail = "Negligible or zero liquidity pool"
        components.append(ScoreComponent(name="Liquidity Depth", score=liq_score, max=25.0, detail=liq_detail))

        # 2. Trading Activity & Balance (20 max)
        vol_24h = (best_pair.volume.get("h24") or 0.0) if best_pair else 0.0
        ratio = best_pair.buy_sell_ratio("h1") if best_pair else None
        if ratio is not None and 0.4 <= ratio <= 2.5 and vol_24h > 10_000:
            vol_score = 20.0
            vol_detail = f"Healthy 24h volume (${vol_24h:,.0f}) and balanced buys/sells"
        elif vol_24h > 2_000:
            vol_score = 12.0
            vol_detail = f"Active volume (${vol_24h:,.0f})"
        else:
            vol_score = 4.0
            vol_detail = "Low trading volume or one-sided orders"
        components.append(ScoreComponent(name="Trading Activity", score=vol_score, max=20.0, detail=vol_detail))

        # 3. Contract Safety & Audit (25 max)
        latest_scan = await self.scans.latest_for_address(chain, address)
        if latest_scan:
            scan_score = max(0.0, 25.0 - (latest_scan.risk_score * 0.25))
            scan_detail = f"Risk score {latest_scan.risk_score}/100, verified={latest_scan.verified}"
        else:
            scan_score = 12.0
            scan_detail = "Unscanned contract (neutral baseline)"
        components.append(
            ScoreComponent(name="Contract Safety", score=round(scan_score, 1), max=25.0, detail=scan_detail)
        )

        # 4. Pair Age & Maturity (15 max)
        age_hours = best_pair.age_hours if best_pair else None
        if age_hours is not None:
            if age_hours >= 720:  # 30+ days
                age_score = 15.0
                age_detail = f"Established token ({round(age_hours / 24)} days old)"
            elif age_hours >= 168:  # 7 days
                age_score = 10.0
                age_detail = f"Maturing token ({round(age_hours / 24)} days old)"
            elif age_hours >= 24:
                age_score = 5.0
                age_detail = "Young token (< 7 days)"
            else:
                age_score = 1.0
                age_detail = "Newly launched pair (< 24 hours)"
        else:
            age_score = 5.0
            age_detail = "Unknown launch date"
        components.append(ScoreComponent(name="Pair Maturity", score=age_score, max=15.0, detail=age_detail))

        # 5. AML & Blacklist Status (15 max)
        bl_entry = await self.blacklist.get_by_address(address)
        if bl_entry and bl_entry.status == BlacklistStatus.BLOCKED:
            aml_score = 0.0
            aml_detail = f"Flagged on AML blacklist: {bl_entry.reason}"
        else:
            aml_score = 15.0
            aml_detail = "Clean AML record"
        components.append(ScoreComponent(name="AML Compliance", score=aml_score, max=15.0, detail=aml_detail))

        total_score = round(sum(c.score for c in components))
        total_score = max(0, min(100, total_score))

        if total_score >= 80:
            label = ScoreLabel.EXCELLENT
        elif total_score >= 65:
            label = ScoreLabel.GOOD
        elif total_score >= 50:
            label = ScoreLabel.FAIR
        elif total_score >= 35:
            label = ScoreLabel.POOR
        else:
            label = ScoreLabel.DANGEROUS

        project_score = ProjectScore(
            chain=chain,
            address=address,
            symbol=symbol or (best_pair.base_symbol if best_pair else None),
            score=total_score,
            label=label,
            components=components,
            inputs={"liquidity_usd": liq_usd, "volume_24h": vol_24h, "pair_age_hours": age_hours},
            computed_at=utcnow(),
        )
        return await self.scores.upsert_score(project_score)

    async def get_project_score(self, chain: str, address: str) -> ProjectScore:
        chain = chain.lower()
        address = address.lower()
        existing = await self.scores.get_for(chain, address)
        if existing and (utcnow() - existing.computed_at).total_seconds() < 3600:
            return existing
        return await self.compute_project_score(chain, address)

    # ----------------------------------------------------------- fraud alerts
    async def create_alert(self, alert: FraudAlert) -> FraudAlert | None:
        inserted = await self.alerts.insert_if_new(alert)
        if inserted:
            logger.info("fraud_alert_created", id=inserted.id, type=inserted.type, severity=inserted.severity)
            event_bus.publish(Topics.ALERT, inserted.model_dump(mode="json"))
            sev = Severity.CRITICAL if inserted.severity == AlertSeverity.CRITICAL else Severity.WARNING
            await self.notifications.notify_subscribers(
                type=NotificationType.FRAUD_ALERT,
                title=inserted.title,
                body=inserted.body,
                severity=sev,
                data={"alert_id": inserted.id, "type": inserted.type, "chain": inserted.chain},
            )
        return inserted

    async def list_alerts(
        self,
        *,
        type: AlertType | None = None,
        severity: AlertSeverity | None = None,
        acknowledged: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FraudAlert], int]:
        query: dict[str, Any] = {}
        if type:
            query["type"] = type
        if severity:
            query["severity"] = severity
        if acknowledged is True:
            query["acknowledged_at"] = {"$ne": None}
        elif acknowledged is False:
            query["acknowledged_at"] = None

        total = await self.alerts.count(query)
        items = await self.alerts.find(query, sort=[("created_at", -1)], limit=limit, skip=offset)
        return items, total

    async def acknowledge_alert(self, alert_id: str, actor: str, note: str | None = None) -> FraudAlert:
        alert = await self.alerts.get_or_raise(alert_id)
        alert.acknowledged_at = utcnow()
        alert.acknowledged_by = actor
        if note:
            alert.mitigation = {"by": actor, "at": utcnow().isoformat(), "note": note}
        return await self.alerts.replace(alert)

    # ----------------------------------------------------------- watched pairs & DEX monitoring
    async def list_watched_pairs(self, active_only: bool = True) -> list[WatchedPair]:
        if active_only:
            return await self.pairs.active()
        return await self.pairs.find({}, sort=[("created_at", -1)])

    async def add_watched_pair(self, payload: WatchedPairIn, added_by: str | None = None) -> WatchedPair:
        chain = payload.chain.lower()
        pair_addr = payload.pair_address.lower()
        data = payload.model_dump(exclude_unset=True)
        data.pop("chain", None)
        data.pop("pair_address", None)
        data["added_by"] = added_by
        pair = await self.pairs.upsert_pair(chain, pair_addr, **data)

        # Poll initial state
        try:
            pair_data = await self.dexscreener.pair_by_address(chain, pair_addr)
            if pair_data:
                await self.pairs.update_where(
                    {"_id": pair.id},
                    {
                        "last_liquidity_usd": pair_data.liquidity_usd,
                        "last_price_usd": pair_data.price_usd,
                        "base_symbol": pair_data.base_symbol or pair.base_symbol,
                        "quote_symbol": pair_data.quote_symbol or pair.quote_symbol,
                        "dex": pair_data.dex or pair.dex,
                        "pair_created_at": pair_data.created_at,
                        "last_checked_at": utcnow(),
                    },
                )
        except Exception as exc:
            logger.warning("initial_pair_fetch_failed", pair=pair_addr, error=str(exc))

        return await self.pairs.get_pair(chain, pair_addr) or pair

    async def check_pair_for_fraud(self, pair: WatchedPair) -> list[FraudAlert]:
        generated: list[FraudAlert] = []
        try:
            dex_pair = await self.dexscreener.pair_by_address(pair.chain, pair.pair_address)
            if not dex_pair or dex_pair.liquidity_usd is None:
                return generated

            now = utcnow()
            # Record snapshot
            await self.snapshots.append(
                {
                    "ts": now,
                    "meta": {"chain": pair.chain, "pair_address": pair.pair_address.lower()},
                    "liquidity_usd": dex_pair.liquidity_usd,
                    "price_usd": dex_pair.price_usd,
                    "volume_h1": dex_pair.volume.get("h1"),
                    "volume_h24": dex_pair.volume.get("h24"),
                    "buys_h1": dex_pair.txns.get("h1", {}).get("buys", 0),
                    "sells_h1": dex_pair.txns.get("h1", {}).get("sells", 0),
                }
            )

            last_liq = pair.last_liquidity_usd
            cur_liq = dex_pair.liquidity_usd

            # Rug pull detection: severe liquidity drain
            if last_liq and last_liq > 5000:
                drop_pct = ((last_liq - cur_liq) / last_liq) * 100.0
                if drop_pct >= self.settings.rug_pull_liquidity_drop_pct:
                    fingerprint = f"rug_pull:{pair.chain}:{pair.pair_address}:{now.strftime('%Y%m%d%H')}"
                    alert = FraudAlert(
                        type=AlertType.RUG_PULL,
                        severity=AlertSeverity.CRITICAL,
                        token_address=pair.token_address,
                        pair_address=pair.pair_address,
                        chain=pair.chain,
                        symbol=pair.base_symbol,
                        title=f"Rug Pull Alert: {pair.base_symbol or pair.pair_address[:10]}",
                        body=f"Liquidity pool drained by {drop_pct:.1f}% from ${last_liq:,.0f} to ${cur_liq:,.0f}.",
                        evidence={
                            "previous_liquidity_usd": last_liq,
                            "current_liquidity_usd": cur_liq,
                            "drop_pct": drop_pct,
                            "threshold_pct": self.settings.rug_pull_liquidity_drop_pct,
                        },
                        fingerprint=fingerprint,
                        risk_score=95,
                    )
                    saved = await self.create_alert(alert)
                    if saved:
                        generated.append(saved)

            # Pump and dump detection: price surge + abnormal volume followed by dumping
            price_change_h1 = dex_pair.price_change.get("h1", 0.0)
            vol_h1 = dex_pair.volume.get("h1", 0.0)
            vol_h24 = dex_pair.volume.get("h24", 0.0)
            expected_h1_vol = (vol_h24 / 24.0) if vol_h24 > 0 else 0.0

            if (
                price_change_h1 >= self.settings.pump_price_spike_pct
                and expected_h1_vol > 0
                and vol_h1 >= (expected_h1_vol * self.settings.pump_volume_multiplier)
            ):
                fingerprint = f"pump_dump:{pair.chain}:{pair.pair_address}:{now.strftime('%Y%m%d%H')}"
                alert = FraudAlert(
                    type=AlertType.PUMP_AND_DUMP,
                    severity=AlertSeverity.WARNING,
                    token_address=pair.token_address,
                    pair_address=pair.pair_address,
                    chain=pair.chain,
                    symbol=pair.base_symbol,
                    title=f"Pump & Dump Detected: {pair.base_symbol or pair.pair_address[:10]}",
                    body=f"Unusual surge detected: +{price_change_h1:.1f}% price spike in 1h with ${vol_h1:,.0f} volume ({vol_h1 / max(expected_h1_vol, 1):.1f}x baseline).",
                    evidence={
                        "price_change_1h_pct": price_change_h1,
                        "volume_1h_usd": vol_h1,
                        "baseline_1h_usd": expected_h1_vol,
                        "buy_sell_ratio": dex_pair.buy_sell_ratio("h1"),
                    },
                    fingerprint=fingerprint,
                    risk_score=75,
                )
                saved = await self.create_alert(alert)
                if saved:
                    generated.append(saved)

            # Update pair stats
            await self.pairs.update_where(
                {"_id": pair.id},
                {
                    "last_liquidity_usd": cur_liq,
                    "last_price_usd": dex_pair.price_usd,
                    "last_checked_at": now,
                },
            )
        except Exception as exc:
            logger.warning("pair_fraud_check_failed", pair=pair.pair_address, error=str(exc))

        return generated

    # ----------------------------------------------------------- AML blacklist
    async def list_blacklist(self, status: BlacklistStatus | None = None, limit: int = 100) -> list[BlacklistEntry]:
        query = {"status": status} if status else {}
        return await self.blacklist.find(query, sort=[("created_at", -1)], limit=limit)

    async def add_to_blacklist(self, payload: BlacklistIn, actor: str | None = None) -> BlacklistEntry:
        addr = payload.address.lower()
        existing = await self.blacklist.get_by_address(addr)
        if existing:
            return (
                await self.blacklist.touch(
                    addr,
                    reason=payload.reason,
                    risk_score=payload.risk_score,
                    status=payload.status,
                    notes=payload.notes,
                    tags=payload.tags,
                )
                or existing
            )

        entry = BlacklistEntry(
            address=addr,
            chain=payload.chain.lower(),
            risk_score=payload.risk_score,
            reason=payload.reason,
            status=payload.status,
            source=payload.source,
            added_by=actor,
            notes=payload.notes,
            tags=payload.tags,
        )
        return await self.blacklist.insert(entry)

    async def check_blacklist(self, address: str) -> BlacklistEntry | None:
        return await self.blacklist.get_by_address(address.lower())

    # ----------------------------------------------------------- stats
    async def stats(self) -> dict[str, Any]:
        since_24h = utcnow() - timedelta(hours=24)
        total_scans = await self.scans.count({})
        recent_scans = await self.scans.count_since(since_24h)
        alert_counts = await self.alerts.counts_by_severity(since_24h)
        monitored_pairs = await self.pairs.count_active()
        blacklisted = await self.blacklist.count({"status": BlacklistStatus.BLOCKED})
        avg_score = await self.scores.average()

        return {
            "total_scans": total_scans,
            "recent_scans_24h": recent_scans,
            "alerts_by_severity": alert_counts,
            "monitored_pairs": monitored_pairs,
            "blacklisted_addresses": blacklisted,
            "average_project_score": avg_score,
        }


__all__ = ["FraudService"]
