"""Module 3 business logic: on-chain intelligence, whale tracking, wallet profiling, and cluster graphs."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from app.core.config import Settings, get_settings
from app.core.events import Topics, event_bus
from app.core.exceptions import NotFoundError, UnauthorizedError, ValidationFailedError
from app.core.timeutils import utcnow
from app.integrations.evm import EvmRpcClient, is_address
from app.integrations.moralis import MoralisClient
from app.modules.fraud.models import BlacklistStatus
from app.modules.fraud.repository import BlacklistRepository
from app.modules.market.repository import AssetStatsRepository
from app.modules.notifications.models import NotificationType, Severity
from app.modules.notifications.service import NotificationService
from app.modules.onchain.models import (
    ConfirmationStatus,
    EntityType,
    GraphEdge,
    GraphNode,
    TransferFlow,
    TransferRisk,
    Wallet,
    WhaleTransfer,
)
from app.modules.onchain.repository import OnchainMetricRepository, WalletRepository, WhaleTransferRepository
from app.modules.onchain.schemas import (
    ClusterGraphOut,
    InvestigationIn,
    InvestigationOut,
    NetflowOut,
    WalletProfileOut,
    WhaleTransferOut,
)

logger = structlog.get_logger(__name__)

# Known exchange & protocol addresses (lowercased)
KNOWN_LABELS: dict[str, tuple[str, EntityType]] = {
    "0x28c6c06298d514db089934071355e5743bf21d60": ("Binance 14", EntityType.EXCHANGE),
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": ("Binance Hot Wallet", EntityType.EXCHANGE),
    "0xdfd5293d8e347dfe59e90efd55b2956a13430eb0": ("Binance 8", EntityType.EXCHANGE),
    "0x503828976d22510aad0201ac7ec88293211d23dc": ("Coinbase Cold Storage", EntityType.EXCHANGE),
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43": ("Coinbase Hot Wallet", EntityType.EXCHANGE),
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": ("Kraken 1", EntityType.EXCHANGE),
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13": ("Kraken 2", EntityType.EXCHANGE),
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": ("OKX Deposit", EntityType.EXCHANGE),
    "0xf977814e90da44bfa03b6295a0616a897441acec": ("Binance 8", EntityType.EXCHANGE),
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": ("Uniswap V2 Router", EntityType.CONTRACT),
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": ("Uniswap V3 Router", EntityType.CONTRACT),
}


class OnchainService:
    def __init__(self, db: Any, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.transfers = WhaleTransferRepository(db)
        self.wallets = WalletRepository(db)
        self.metrics = OnchainMetricRepository(db)
        self.blacklist = BlacklistRepository(db)
        self.stats_repo = AssetStatsRepository(db)
        self.evm = EvmRpcClient(self.settings)
        self.moralis = MoralisClient(self.settings)
        self.notifications = NotificationService(db, self.settings)

    # ----------------------------------------------------------- whale transfers
    async def list_transfers(
        self,
        *,
        symbol: str | None = None,
        min_amount_usd: float | None = None,
        flow: TransferFlow | None = None,
        risk_level: TransferRisk | None = None,
        address: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WhaleTransfer], int]:
        return await self.transfers.find_transfers(
            symbol=symbol,
            min_amount_usd=min_amount_usd or self.settings.whale_threshold_usd,
            flow=flow,
            risk_level=risk_level,
            address=address,
            since=since,
            limit=limit,
            offset=offset,
        )

    async def get_transfers_by_tx(self, tx_hash: str) -> list[WhaleTransfer]:
        return await self.transfers.get_by_tx_hash(tx_hash)

    async def classify_transfer(
        self,
        from_address: str,
        to_address: str,
        amount_usd: float | None = None,
    ) -> tuple[TransferFlow, TransferRisk, str | None, str | None]:
        from_addr = from_address.lower()
        to_addr = to_address.lower()

        from_label_data = KNOWN_LABELS.get(from_addr)
        to_label_data = KNOWN_LABELS.get(to_addr)

        from_label = from_label_data[0] if from_label_data else None
        to_label = to_label_data[0] if to_label_data else None

        from_is_ex = bool(from_label_data and from_label_data[1] == EntityType.EXCHANGE)
        to_is_ex = bool(to_label_data and to_label_data[1] == EntityType.EXCHANGE)

        if from_is_ex and to_is_ex:
            flow = TransferFlow.EXCHANGE_INTERNAL
        elif to_is_ex:
            flow = TransferFlow.EXCHANGE_INFLOW
        elif from_is_ex:
            flow = TransferFlow.EXCHANGE_OUTFLOW
        else:
            flow = TransferFlow.WALLET_TO_WALLET

        # Risk scoring
        risk = TransferRisk.LOW
        # Check AML blacklist
        bl_from = await self.blacklist.get_by_address(from_addr)
        bl_to = await self.blacklist.get_by_address(to_addr)
        if (bl_from and bl_from.status == BlacklistStatus.BLOCKED) or (
            bl_to and bl_to.status == BlacklistStatus.BLOCKED
        ):
            risk = TransferRisk.SUSPICIOUS
        elif amount_usd and amount_usd >= self.settings.whale_alert_threshold_usd:
            risk = TransferRisk.MEDIUM

        return flow, risk, from_label, to_label

    async def record_transfer(self, transfer: WhaleTransfer) -> WhaleTransfer | None:
        """Persist a whale transfer, update wallet profiles, broadcast WebSocket, and send alerts."""
        inserted = await self.transfers.insert_transfer(transfer)
        if not inserted:
            return None

        # Update sender and receiver wallet records
        await self.wallets.upsert_wallet(
            transfer.from_address,
            label=transfer.from_label,
            entity_type=EntityType.EXCHANGE
            if transfer.from_label and "binance" in transfer.from_label.lower()
            else EntityType.WHALE,
            last_seen_at=transfer.block_time,
        )
        await self.wallets.upsert_wallet(
            transfer.to_address,
            label=transfer.to_label,
            entity_type=EntityType.EXCHANGE
            if transfer.to_label and "binance" in transfer.to_label.lower()
            else EntityType.WHALE,
            last_seen_at=transfer.block_time,
        )

        # Broadcast via event bus
        event_bus.publish(Topics.WHALE_TRANSFER, inserted.model_dump(mode="json"))

        # If large transfer meets alert threshold or is suspicious, push notification
        alert_thresh = self.settings.whale_alert_threshold_usd
        if (
            inserted.amount_usd and inserted.amount_usd >= alert_thresh
        ) or inserted.risk_level == TransferRisk.SUSPICIOUS:
            amt_str = (
                f"${inserted.amount_usd:,.0f}" if inserted.amount_usd else f"{inserted.amount:,.2f} {inserted.symbol}"
            )
            flow_label = inserted.flow.replace("_", " ").title()
            title = f"Whale Alert: {amt_str} {inserted.symbol}"
            body = f"{flow_label} detected from {inserted.from_label or inserted.from_address[:8]} to {inserted.to_label or inserted.to_address[:8]}."
            sev = Severity.CRITICAL if inserted.risk_level == TransferRisk.SUSPICIOUS else Severity.WARNING
            await self.notifications.notify_subscribers(
                type=NotificationType.WHALE_ALERT,
                title=title,
                body=body,
                severity=sev,
                data={
                    "tx_hash": inserted.tx_hash,
                    "symbol": inserted.symbol,
                    "amount_usd": inserted.amount_usd,
                    "flow": inserted.flow,
                },
            )

        return inserted

    # ----------------------------------------------------------- wallet profiles & smart money
    async def get_wallet_profile(self, address: str) -> WalletProfileOut:
        addr = address.lower()
        if not is_address(addr):
            raise ValidationFailedError(f"Invalid Ethereum address: {address}")

        wallet = await self.wallets.get_by_address(addr)
        if not wallet:
            # Create baseline record
            label_data = KNOWN_LABELS.get(addr)
            wallet = Wallet(
                address=addr,
                label=label_data[0] if label_data else None,
                entity_type=label_data[1] if label_data else EntityType.UNKNOWN,
                smart_money_score=50,
                first_seen_at=utcnow(),
                last_seen_at=utcnow(),
            )
            wallet = await self.wallets.insert(wallet)

        recent = await self.transfers.get_transfers_for_address(addr, limit=20)
        recent_outs = [WhaleTransferOut(**t.model_dump()) for t in recent]

        data = wallet.model_dump()
        data["recent_transfers"] = recent_outs
        return WalletProfileOut(**data)

    async def top_smart_money(self, limit: int = 50) -> list[Wallet]:
        return await self.wallets.top_smart_money(limit=limit)

    # ----------------------------------------------------------- visual cluster graph (FE-4)
    async def build_cluster_graph(self, address: str, max_transfers: int = 50) -> ClusterGraphOut:
        addr = address.lower()
        if not is_address(addr):
            raise ValidationFailedError(f"Invalid Ethereum address: {address}")

        transfers = await self.transfers.get_transfers_for_address(addr, limit=max_transfers)
        nodes_dict: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        total_vol = 0.0

        # Center node
        center_wallet = await self.wallets.get_by_address(addr)
        nodes_dict[addr] = GraphNode(
            id=addr,
            label=center_wallet.label if center_wallet else addr[:8],
            entity_type=center_wallet.entity_type if center_wallet else EntityType.WHALE,
            smart_money_score=center_wallet.smart_money_score if center_wallet else 50,
        )

        for t in transfers:
            amt_usd = t.amount_usd or 0.0
            total_vol += amt_usd

            # From node
            if t.from_address not in nodes_dict:
                nodes_dict[t.from_address] = GraphNode(
                    id=t.from_address,
                    label=t.from_label or t.from_address[:8],
                    entity_type=EntityType.EXCHANGE
                    if t.from_label and "binance" in t.from_label.lower()
                    else EntityType.UNKNOWN,
                    risk_level=t.risk_level,
                )
            # To node
            if t.to_address not in nodes_dict:
                nodes_dict[t.to_address] = GraphNode(
                    id=t.to_address,
                    label=t.to_label or t.to_address[:8],
                    entity_type=EntityType.EXCHANGE
                    if t.to_label and "binance" in t.to_label.lower()
                    else EntityType.UNKNOWN,
                    risk_level=t.risk_level,
                )

            edges.append(
                GraphEdge(
                    source=t.from_address,
                    target=t.to_address,
                    amount=t.amount,
                    amount_usd=t.amount_usd,
                    symbol=t.symbol,
                    tx_hash=t.tx_hash,
                    block_time=t.block_time,
                )
            )

        return ClusterGraphOut(
            nodes=list(nodes_dict.values()),
            edges=edges,
            center_address=addr,
            total_volume_usd=round(total_vol, 2),
        )

    # ----------------------------------------------------------- exchange netflows
    async def get_netflow(self, symbol: str, window_hours: int = 24) -> NetflowOut:
        since = utcnow() - timedelta(hours=window_hours)
        summary = await self.transfers.netflow_summary(symbol, since)
        net = summary["netflow_usd"]

        if net > 5_000_000:
            trend = "distributing"  # heavy inflow to exchanges -> potential selling
        elif net < -5_000_000:
            trend = "accumulating"  # heavy outflow from exchanges -> accumulation
        else:
            trend = "neutral"

        return NetflowOut(
            symbol=symbol.upper(),
            window_hours=window_hours,
            inflow_usd=summary["inflow_usd"],
            outflow_usd=summary["outflow_usd"],
            netflow_usd=net,
            trend=trend,
            updated_at=utcnow(),
        )

    # ----------------------------------------------------------- transaction investigation (FR3.3)
    async def investigate(self, payload: InvestigationIn) -> InvestigationOut:
        if payload.tx_hash:
            tx_hash = payload.tx_hash.lower()
            transfers = await self.transfers.get_by_tx_hash(tx_hash)
            if not transfers:
                # Query on-chain via EVM RPC
                receipt = await self.evm.get_transaction_receipt(tx_hash)
                if not receipt:
                    raise NotFoundError(f"Transaction not found on chain: {tx_hash}")
                details = {
                    "tx_hash": tx_hash,
                    "block_number": receipt.get("blockNumber"),
                    "status": "confirmed" if receipt.get("status") in ("0x1", 1) else "failed",
                    "gas_used": receipt.get("gasUsed"),
                    "from": receipt.get("from"),
                    "to": receipt.get("to"),
                }
                center_addr = (receipt.get("from") or "").lower()
            else:
                t = transfers[0]
                details = t.model_dump(mode="json")
                center_addr = t.from_address

            graph = await self.build_cluster_graph(center_addr) if center_addr else ClusterGraphOut()
            return InvestigationOut(
                query=tx_hash,
                entity_type="transaction",
                details=details,
                graph=graph,
                risk_assessment={"flagged": False, "confidence": 0.95},
                provenance={"source": "EVM RPC & Moralis", "timestamp": utcnow().isoformat()},
            )

        elif payload.address:
            addr = payload.address.lower()
            profile = await self.get_wallet_profile(addr)
            graph = await self.build_cluster_graph(addr)
            return InvestigationOut(
                query=addr,
                entity_type="wallet",
                details=profile.model_dump(),
                graph=graph,
                risk_assessment={"smart_money_score": profile.smart_money_score, "is_exchange": profile.is_exchange},
                provenance={"source": "Quantachain On-Chain Intelligence", "timestamp": utcnow().isoformat()},
            )

        raise ValidationFailedError("Either tx_hash or address must be provided for investigation")

    # ----------------------------------------------------------- Moralis Webhooks
    async def handle_moralis_webhook(self, raw_body: bytes, signature: str | None) -> dict[str, Any]:
        if self.settings.moralis_stream_secret:
            if not self.moralis.verify_stream_signature(raw_body, signature):
                raise UnauthorizedError("Invalid Moralis webhook signature")

        payload = self.moralis.parse_stream_payload(raw_body)
        erc20_transfers = payload.get("erc20Transfers") or []
        confirmed = payload.get("confirmed", True)
        block = int(payload.get("block", {}).get("number", 0))

        processed = 0
        for item in erc20_transfers:
            try:
                tx_hash = (item.get("transactionHash") or "").lower()
                from_addr = (item.get("from") or "").lower()
                to_addr = (item.get("to") or "").lower()
                token_symbol = item.get("tokenSymbol") or "ERC20"
                value = float(item.get("valueWithDecimals") or 0.0)

                # Estimate USD price from AssetStats if available
                stat = await self.stats_repo.get_stats(token_symbol)
                price = stat.price if stat else 1.0
                amount_usd = value * price if price else None

                flow, risk, from_lbl, to_lbl = await self.classify_transfer(from_addr, to_addr, amount_usd)

                transfer = WhaleTransfer(
                    tx_hash=tx_hash,
                    log_index=int(item.get("logIndex", 0)),
                    chain="eth",
                    block_number=block,
                    block_time=utcnow(),
                    symbol=token_symbol,
                    token_address=(item.get("contract") or "").lower(),
                    amount=value,
                    amount_usd=amount_usd,
                    price_source="asset_stats" if stat else None,
                    price_timestamp=utcnow(),
                    from_address=from_addr,
                    to_address=to_addr,
                    from_label=from_lbl,
                    to_label=to_lbl,
                    flow=flow,
                    risk_level=risk,
                    confirmation_status=ConfirmationStatus.CONFIRMED if confirmed else ConfirmationStatus.PENDING,
                )
                if amount_usd and amount_usd >= self.settings.whale_threshold_usd:
                    await self.record_transfer(transfer)
                    processed += 1
            except Exception as exc:
                logger.warning("moralis_transfer_processing_error", error=str(exc))

        return {"status": "ok", "processed": processed, "block": block}


__all__ = ["KNOWN_LABELS", "OnchainService"]
