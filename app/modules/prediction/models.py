"""Module 5 persistence models: predictions, volatility forecasts, trade signals, ML models, reports.

``Report`` and ``ReportRepository`` (``repository.py``) are shared with Module 8, which
produces other report types with the same document shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.db.models import Document


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class SignalAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class VolRegime(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class ReportStatus(StrEnum):
    QUEUED = "queued"
    READY = "ready"
    FAILED = "failed"


class ReportFormat(StrEnum):
    PDF = "pdf"
    CSV = "csv"


class ReportType(StrEnum):
    """Well-known report types. ``Report.type`` is a plain string so other modules can add kinds."""

    TOKEN_FORECAST = "token_forecast"
    MARKET_RESEARCH = "market_research"
    PORTFOLIO = "portfolio"
    WHALE_ACTIVITY = "whale_activity"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


class Prediction(Document):
    """One price forecast for ``symbol`` over ``horizon``; a new document is written on every refresh.

    ``latest`` marks the most recent prediction per symbol/horizon. Once ``target_time`` has
    passed the evaluation job fills ``realized_price``, ``abs_error_pct`` and ``direction_hit``.
    """

    symbol: str
    horizon: str  # 1h | 4h | 24h (settings.prediction_horizons)
    interval: str = "1h"  # candle interval the model was fitted on
    steps: int  # forecast steps on ``interval`` candles
    price: float  # price when the forecast was made
    target_price: float
    lower: float
    upper: float
    change_pct: float
    direction: Direction
    confidence: float = Field(ge=0, le=1)
    risk_reward: float | None = None
    stop_distance: float | None = None
    flat_threshold_pct: float = 0.0  # |move| below this counts as "flat"
    target_time: datetime
    model: str  # statistical | lstm
    technical_target: float | None = None  # before the sentiment / whale adjustment
    adjustment_pct: float | None = None
    factors: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    feature_snapshot: dict[str, float | None] = Field(default_factory=dict)
    path_points: list[float] = Field(default_factory=list)  # hourly path (24h horizon only)
    path_lower: list[float] = Field(default_factory=list)
    path_upper: list[float] = Field(default_factory=list)
    samples: int = 0
    latest: bool = True
    evaluated: bool = False
    realized_price: float | None = None
    abs_error_pct: float | None = None
    direction_hit: bool | None = None
    evaluated_at: datetime | None = None


class VolatilityForecast(Document):
    """Volatility estimators, next-24h forecast and regime classification for one symbol."""

    symbol: str
    interval: str = "1h"
    realized_24h_pct: float | None = None  # annualised close-to-close, last 24 candles
    realized_7d_pct: float | None = None
    parkinson_24h_pct: float | None = None
    ewma_pct: float | None = None
    garch_pct: float | None = None
    forecast_24h_annualised_pct: float | None = None
    forecast_24h_horizon_pct: float | None = None  # expected 1-sigma move over the next 24h
    horizon_sigmas: list[float] = Field(default_factory=list)  # cumulative hourly sigma (decimals), k = 1..24
    regime: VolRegime = VolRegime.NORMAL
    percentile: float | None = None  # of current vol within the historic volatility index window
    observations: int = 0  # daily observations behind the percentile
    history_times: list[datetime] = Field(default_factory=list)
    history_values_pct: list[float] = Field(default_factory=list)
    params: dict[str, float] = Field(default_factory=dict)  # ewma_lambda, omega, alpha, beta, ...
    latest: bool = True


class TradeSignal(Document):
    symbol: str
    horizon: str
    action: SignalAction
    direction: Direction
    strength: float  # z = expected move / forecast volatility over the horizon
    confidence: float = Field(ge=0, le=1)
    entry_price: float
    target_price: float
    stop_loss: float
    expected_move_pct: float
    sigma_pct: float
    sigma_threshold: float
    rationale: str
    expires_at: datetime
    prediction_id: str | None = None
    model: str | None = None


class MLModel(Document):
    name: str  # lstm
    symbol: str
    interval: str
    version: int = 1
    framework: str = "tensorflow"
    trained_at: datetime
    samples: int
    epochs: int
    lookback: int
    features: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)  # directional_accuracy, mape, rmse
    scaler: dict[str, Any] = Field(default_factory=dict)  # standardisation constants used at inference
    path: str
    active: bool = True


class Report(Document):
    """Generated document (PDF / CSV) owned by a user. Shared with Module 8."""

    user_id: str
    type: str  # ReportType values or other module-specific kinds
    title: str
    symbol: str | None = None
    status: ReportStatus = ReportStatus.QUEUED
    format: ReportFormat = ReportFormat.PDF
    file_path: str | None = None
    size_bytes: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    completed_at: datetime | None = None


__all__ = [
    "Direction",
    "MLModel",
    "Prediction",
    "Report",
    "ReportFormat",
    "ReportStatus",
    "ReportType",
    "SignalAction",
    "TradeSignal",
    "VolRegime",
    "VolatilityForecast",
]
