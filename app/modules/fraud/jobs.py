"""Background jobs for Module 6: DEX pair liquidity monitoring & fraud detection."""

from __future__ import annotations

import structlog

from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec

logger = structlog.get_logger(__name__)


def register(runtime: WorkerRuntime) -> None:
    async def monitor_dex_pairs() -> None:
        from app.modules.fraud.service import FraudService

        service = FraudService(runtime.db, runtime.settings)
        pairs = await service.list_watched_pairs(active_only=True)
        if not pairs:
            return

        total_alerts = 0
        for pair in pairs:
            try:
                alerts = await service.check_pair_for_fraud(pair)
                total_alerts += len(alerts)
            except Exception as exc:
                logger.warning("dex_pair_check_failed", pair=pair.pair_address, error=str(exc))

        if total_alerts > 0:
            logger.info("dex_fraud_monitoring_cycle_completed", total_pairs=len(pairs), new_alerts=total_alerts)

    runtime.add_job(
        JobSpec(
            name="fraud.monitor_dex_pairs",
            interval_seconds=runtime.settings.fraud_poll_seconds,
            func=monitor_dex_pairs,
            module="fraud",
            description="Poll monitored DEX pairs and detect rug pulls and pump-and-dump spikes",
            run_on_start=False,
            timeout_seconds=60.0,
        )
    )
