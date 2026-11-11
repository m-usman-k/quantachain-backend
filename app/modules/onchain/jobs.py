"""Background jobs for Module 3: EVM block scanning & on-chain netflow updates."""

from __future__ import annotations

import structlog

from app.core.timeutils import utcnow
from app.integrations.evm import hex_to_int
from app.modules.onchain.models import ConfirmationStatus, WhaleTransfer
from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec

logger = structlog.get_logger(__name__)


def register(runtime: WorkerRuntime) -> None:
    async def scan_onchain_blocks() -> None:
        from app.modules.onchain.service import OnchainService

        service = OnchainService(runtime.db, runtime.settings)
        try:
            latest_block = await service.evm.block_number()
            if not latest_block:
                return

            # Scan the latest block
            block_data = await service.evm.get_block_by_number(latest_block, full_txs=True)
            if not block_data:
                return

            txs = block_data.get("transactions") or []
            eth_price = 3000.0
            stat = await service.stats_repo.get_stats("ETH")
            if stat and stat.price:
                eth_price = stat.price

            # Look for large ETH transfers (> 100 ETH)
            for tx in txs:
                if not isinstance(tx, dict):
                    continue
                val_wei = hex_to_int(tx.get("value"))
                val_eth = val_wei / 1e18
                val_usd = val_eth * eth_price

                if val_usd >= runtime.settings.whale_threshold_usd:
                    from_addr = (tx.get("from") or "").lower()
                    to_addr = (tx.get("to") or "").lower()
                    tx_hash = (tx.get("hash") or "").lower()

                    flow, risk, from_lbl, to_lbl = await service.classify_transfer(from_addr, to_addr, val_usd)
                    transfer = WhaleTransfer(
                        tx_hash=tx_hash,
                        log_index=0,
                        chain="eth",
                        block_number=latest_block,
                        block_time=utcnow(),
                        symbol="ETH",
                        amount=round(val_eth, 4),
                        amount_usd=round(val_usd, 2),
                        price_source="asset_stats",
                        price_timestamp=utcnow(),
                        from_address=from_addr,
                        to_address=to_addr,
                        from_label=from_lbl,
                        to_label=to_lbl,
                        flow=flow,
                        risk_level=risk,
                        confirmation_status=ConfirmationStatus.CONFIRMED,
                    )
                    await service.record_transfer(transfer)

        except Exception as exc:
            logger.warning("onchain_block_scan_failed", error=str(exc))

    runtime.add_job(
        JobSpec(
            name="onchain.scan_blocks",
            interval_seconds=runtime.settings.onchain_poll_seconds,
            func=scan_onchain_blocks,
            module="onchain",
            description="Poll latest EVM blocks for large ETH & token whale transfers",
            run_on_start=False,
            timeout_seconds=30.0,
        )
    )
