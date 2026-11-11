"""API schemas for Module 5."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.prediction.models import (
    Direction,
    MLModel,
    Prediction,
    Report,
    SignalAction,
    TradeSignal,
    VolatilityForecast,
    VolRegime,
)


class HorizonOut(BaseModel):
    id: str
    horizon: str
    steps: int
    price: float
    target_price: float
    lower: float
    upper: float
    change_pct: float
    direction: Direction
    confidence: float
    risk_reward: float | None
    stop_distance: float | None
    target_time: datetime
    model: str
    technical_target: float | None
    adjustment_pct: float | None
    factors: dict[str, float]
    created_at: datetime
    evaluated: bool
    realized_price: float | None
    abs_error_pct: float | None
    direction_hit: bool | None

    @classmethod
    def from_model(cls, p: Prediction) -> HorizonOut:
        return cls(
            id=p.id or "",
            horizon=p.horizon,
            steps=p.steps,
            price=p.price,
            target_price=p.target_price,
            lower=p.lower,
            upper=p.upper,
            change_pct=p.change_pct,
            direction=p.direction,
            confidence=p.confidence,
            risk_reward=p.risk_reward,
            stop_distance=p.stop_distance,
            target_time=p.target_time,
            model=p.model,
            technical_target=p.technical_target,
            adjustment_pct=p.adjustment_pct,
            factors=p.factors,
            created_at=p.created_at,
            evaluated=p.evaluated,
            realized_price=p.realized_price,
            abs_error_pct=p.abs_error_pct,
            direction_hit=p.direction_hit,
        )


class MatrixRow(BaseModel):
    symbol: str
    pair: str
    price: float | None
    horizons: list[HorizonOut]
    updated_at: datetime | None


class MatrixOut(BaseModel):
    rows: list[MatrixRow]
    horizons: list[str]
    aggregate_accuracy_pct: float | None
    generated_at: datetime


class PathPoint(BaseModel):
    t: datetime
    point: float
    lower: float
    upper: float


class VolatilityOut(BaseModel):
    symbol: str
    realized_24h_pct: float | None
    realized_7d_pct: float | None
    parkinson_24h_pct: float | None
    ewma_pct: float | None
    garch_pct: float | None
    forecast_24h_annualised_pct: float | None
    forecast_24h_horizon_pct: float | None
    regime: VolRegime
    percentile: float | None
    observations: int
    history: list[dict[str, Any]]
    params: dict[str, float]
    created_at: datetime

    @classmethod
    def from_model(cls, v: VolatilityForecast) -> VolatilityOut:
        return cls(
            symbol=v.symbol,
            realized_24h_pct=v.realized_24h_pct,
            realized_7d_pct=v.realized_7d_pct,
            parkinson_24h_pct=v.parkinson_24h_pct,
            ewma_pct=v.ewma_pct,
            garch_pct=v.garch_pct,
            forecast_24h_annualised_pct=v.forecast_24h_annualised_pct,
            forecast_24h_horizon_pct=v.forecast_24h_horizon_pct,
            regime=v.regime,
            percentile=v.percentile,
            observations=v.observations,
            history=[
                {"t": t, "v": round(val, 2)} for t, val in zip(v.history_times, v.history_values_pct, strict=False)
            ],
            params=v.params,
            created_at=v.created_at,
        )


class SignalOut(BaseModel):
    id: str
    symbol: str
    horizon: str
    action: SignalAction
    direction: Direction
    strength: float
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    expected_move_pct: float
    sigma_pct: float
    rationale: str
    expires_at: datetime
    active: bool
    model: str | None
    created_at: datetime

    @classmethod
    def from_model(cls, s: TradeSignal, *, now: datetime) -> SignalOut:
        return cls(
            id=s.id or "",
            symbol=s.symbol,
            horizon=s.horizon,
            action=s.action,
            direction=s.direction,
            strength=s.strength,
            confidence=s.confidence,
            entry_price=s.entry_price,
            target_price=s.target_price,
            stop_loss=s.stop_loss,
            expected_move_pct=s.expected_move_pct,
            sigma_pct=s.sigma_pct,
            rationale=s.rationale,
            expires_at=s.expires_at,
            active=s.expires_at > now,
            model=s.model,
            created_at=s.created_at,
        )


class ModelInfo(BaseModel):
    forecaster: str
    lstm_available: bool
    lstm_enabled: bool
    lstm_trained: bool
    lstm_metrics: dict[str, float] | None
    trained_at: datetime | None


class AccuracyBucket(BaseModel):
    horizon: str
    count: int
    hits: int
    directional_accuracy_pct: float
    mape_pct: float
    avg_confidence: float


class AccuracyOut(BaseModel):
    overall: AccuracyBucket | None
    by_horizon: list[AccuracyBucket]
    symbol: str | None
    window_days: int


class SymbolForecastOut(BaseModel):
    symbol: str
    pair: str
    price: float | None
    horizons: list[HorizonOut]
    path: list[PathPoint]
    weights: dict[str, float]
    volatility: VolatilityOut | None
    signals: list[SignalOut]
    model: ModelInfo
    accuracy: AccuracyOut
    feature_snapshot: dict[str, float | None]
    generated_at: datetime


class WeightsIn(BaseModel):
    technical: float = Field(ge=0, le=1)
    whale: float = Field(ge=0, le=1)
    sentiment: float = Field(ge=0, le=1)


class WeightsOut(WeightsIn):
    updated_at: datetime | None = None
    updated_by: str | None = None


class MLModelOut(BaseModel):
    id: str
    name: str
    symbol: str
    interval: str
    version: int
    framework: str
    trained_at: datetime
    samples: int
    epochs: int
    lookback: int
    metrics: dict[str, float]
    active: bool

    @classmethod
    def from_model(cls, m: MLModel) -> MLModelOut:
        return cls(
            id=m.id or "",
            name=m.name,
            symbol=m.symbol,
            interval=m.interval,
            version=m.version,
            framework=m.framework,
            trained_at=m.trained_at,
            samples=m.samples,
            epochs=m.epochs,
            lookback=m.lookback,
            metrics=m.metrics,
            active=m.active,
        )


class ModelsStatusOut(BaseModel):
    tensorflow_available: bool
    lstm_enabled: bool
    forecaster_in_use: str
    lookback: int
    horizons: list[str]
    models: list[MLModelOut]
    paused: dict[str, bool]


class TrainRequest(BaseModel):
    epochs: int = Field(default=20, ge=1, le=200)
    interval: str = Field(default="1h", pattern="^(1h)$")


class RefreshResult(BaseModel):
    symbols: list[str]
    skipped: dict[str, str]
    predictions: int
    signals: int
    duration_ms: float


class ReportOut(BaseModel):
    id: str
    type: str
    title: str
    symbol: str | None
    status: str
    format: str
    size_bytes: int | None
    params: dict[str, Any]
    error: str | None
    created_at: datetime
    completed_at: datetime | None
    download_url: str

    @classmethod
    def from_model(cls, r: Report, *, prefix: str) -> ReportOut:
        return cls(
            id=r.id or "",
            type=r.type,
            title=r.title,
            symbol=r.symbol,
            status=str(r.status),
            format=str(r.format),
            size_bytes=r.size_bytes,
            params=r.params,
            error=r.error,
            created_at=r.created_at,
            completed_at=r.completed_at,
            download_url=f"{prefix}/predictions/reports/{r.id}/download",
        )


__all__ = [name for name in globals() if name[0].isupper()]
