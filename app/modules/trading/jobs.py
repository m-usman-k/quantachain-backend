"""Module 9 background work: paper matching, portfolio snapshots and live order sync."""

from __future__ import annotations

from app.modules.trading.paper import SOURCE_NAME, PaperMatcher
from app.modules.trading.portfolio import PortfolioService
from app.modules.trading.service import LiveExecutor, TradingService
from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec


def register(runtime: WorkerRuntime) -> None:
    settings = runtime.settings
    runtime.tracker.register(
        SOURCE_NAME,
        label="Paper Trading Engine",
        kind="engine",
        engine="Trade Execution",
        description="Matches paper orders against live prices",
    )
    matcher = PaperMatcher(runtime.db, settings, tracker=runtime.tracker, bus=runtime.bus)
    portfolio = PortfolioService(runtime.db, settings, runtime.bus)

    runtime.add_stream("paper_matcher", matcher.run, module="trading")
    runtime.add_job(
        JobSpec(
            "paper_order_sweep",
            interval_seconds=15,
            func=matcher.sweep,
            module="trading",
            description="Safety sweep of working paper orders against the latest prices",
            run_on_start=False,
            timeout_seconds=60,
        )
    )
    runtime.add_job(
        JobSpec(
            "portfolio_snapshot",
            interval_seconds=60,
            func=portfolio.snapshot_all,
            module="trading",
            description="Record equity curve points for active paper accounts",
            run_on_start=False,
            timeout_seconds=120,
        )
    )

    if settings.live_trading_enabled:
        executor = LiveExecutor(TradingService(runtime.db, settings, bus=runtime.bus))
        runtime.add_job(
            JobSpec(
                "live_order_sync",
                interval_seconds=30,
                func=executor.sync,
                module="trading",
                description="Reconcile open live orders with the exchange",
                run_on_start=False,
                timeout_seconds=120,
            )
        )
