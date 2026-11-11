"""Module 5 background work: forecast refresh, evaluation and optional LSTM training."""

from __future__ import annotations

from app.core.exceptions import ServiceUnavailableError
from app.modules.market.models import SourceStatus
from app.modules.prediction.forecasters import LSTMForecaster
from app.modules.prediction.service import PredictionService
from app.workers.runtime import WorkerRuntime
from app.workers.scheduler import JobSpec

SOURCE = "prediction_engine"


def register(runtime: WorkerRuntime) -> None:
    settings = runtime.settings
    service = PredictionService(runtime.db, settings, bus=runtime.bus)
    runtime.tracker.register(
        SOURCE,
        label="QuantaAI Predict",
        kind="ai",
        engine="QuantaAI Predict",
        description="LSTM / statistical price forecasts, volatility regimes and trade signals",
    )

    async def refresh() -> dict[str, object]:
        try:
            result = await service.refresh_all()
        except ServiceUnavailableError as exc:
            runtime.tracker.set_status(SOURCE, SourceStatus.STOPPED, message=exc.message)
            return {"skipped": exc.message}
        except Exception as exc:
            runtime.tracker.error(SOURCE, f"{type(exc).__name__}: {exc}")
            raise
        runtime.tracker.event(SOURCE, count=max(result.predictions, 1), latency_ms=result.duration_ms)
        runtime.tracker.set_status(SOURCE, SourceStatus.ACTIVE)
        await runtime.tracker.log(
            SOURCE,
            f"Forecast refresh: {result.predictions} predictions, {result.signals} signals for {len(result.symbols)} symbols in {result.duration_ms:.0f}ms",
        )
        return result.model_dump()

    runtime.add_job(
        JobSpec(
            "prediction_refresh",
            interval_seconds=settings.prediction_refresh_seconds,
            func=refresh,
            module="prediction",
            description="Forecast every tracked symbol, update volatility and emit signals",
            jitter_seconds=30,
            timeout_seconds=900,
        )
    )
    runtime.add_job(
        JobSpec(
            "prediction_evaluate",
            interval_seconds=3600,
            func=service.evaluate_pending,
            module="prediction",
            description="Score matured forecasts against realised prices",
            run_on_start=False,
            timeout_seconds=600,
        )
    )
    runtime.add_job(
        JobSpec(
            "lstm_train",
            interval_seconds=24 * 3600,
            func=service.train_all,
            module="prediction",
            description="Re-train the LSTM forecaster for every tracked symbol (requires TensorFlow)",
            run_on_start=False,
            enabled=settings.lstm_enabled and LSTMForecaster.available(),
            timeout_seconds=3600,
        )
    )
