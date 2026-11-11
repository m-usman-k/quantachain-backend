"""Final-forecast assembly: technical forecast + sentiment / whale adjustment, confidence and risk/reward.

Adjustment (per horizon point)::

    adjusted = point * (1 + w_sent * clip(sentiment_z, -2, 2) * s + w_whale * clip(whale_z, -2, 2) * s)

with ``s = ADJUSTMENT_SCALE = 0.01`` so a +2 sigma reading on either factor moves the
forecast by at most ``2 * s * w`` (0.46% for sentiment, 0.70% for whales at default
weights). ``whale_z = -netflow_z``: exchange inflows (positive net flow) are selling
pressure, outflows (accumulation) are bullish. The nudge is phased in linearly along
the path (the 1h point gets 1/24 of the full-horizon adjustment) because sentiment and
flow effects play out over hours, not instantly. ``technical`` is the residual weight
that keeps the model's own point as the anchor; the three weights must sum to one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, model_validator

from app.core.timeutils import interval_seconds
from app.modules.prediction.forecasters import ForecastResult
from app.modules.prediction.models import Direction

ADJUSTMENT_SCALE = 0.01  # ``s`` in the formula above
Z_CLIP = 2.0
NETFLOW_SIGN = -1.0
FLAT_SIGMA_FRACTION = 0.1  # |expected move| below this fraction of the horizon sigma is "flat"
STOP_ATR_MULTIPLIER = 1.5  # stop distance = 1.5 * ATR(14) * sqrt(steps)
MIN_CANDLES = 200  # symbols with fewer 1h candles are skipped
FULL_DATA_CANDLES = 720  # 30 days of hourly candles earns the full data-sufficiency score
ACCURACY_PRIOR_WEIGHT = 10  # pseudo-evaluations at 50% accuracy blended into young models
CONFIDENCE_WEIGHTS = (0.45, 0.35, 0.20)  # interval tightness, directional accuracy, data sufficiency
CONFIDENCE_BOUNDS = (0.05, 0.95)
INTERVAL_SHARPNESS = 10.0  # interval score = 1 / (1 + 10 * relative band width)
WEIGHT_TOLERANCE = 0.01


class SignalWeights(BaseModel):
    """Blend weights shown in the UI and used by the adjustment; admins edit them via ``PUT /predictions/weights``."""

    technical: float = Field(default=0.42, ge=0, le=1)
    whale: float = Field(default=0.35, ge=0, le=1)
    sentiment: float = Field(default=0.23, ge=0, le=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> SignalWeights:
        total = self.technical + self.whale + self.sentiment
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"weights must sum to 1 (+/- {WEIGHT_TOLERANCE}); got {total:.3f}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {"technical": self.technical, "whale": self.whale, "sentiment": self.sentiment}


DEFAULT_WEIGHTS = SignalWeights()


def horizon_steps(horizon: str, base_interval: str = "1h") -> int:
    """Number of ``base_interval`` candles in ``horizon`` (``24h`` on 1h candles -> 24)."""
    seconds = interval_seconds(horizon)
    base = interval_seconds(base_interval)
    if seconds % base:
        raise ValueError(f"Horizon {horizon} is not a multiple of {base_interval}")
    return max(1, seconds // base)


def clip_z(value: float | None) -> float:
    if value is None or math.isnan(value):
        return 0.0
    return float(min(Z_CLIP, max(-Z_CLIP, value)))


@dataclass(slots=True)
class Adjustment:
    sentiment_z: float
    whale_z: float
    sentiment_pct: float  # contribution in percent of price at the full horizon
    whale_pct: float
    total: float  # fractional adjustment at the full horizon

    def as_factors(self) -> dict[str, float]:
        return {
            "sentiment_z": round(self.sentiment_z, 4),
            "whale_z": round(self.whale_z, 4),
            "sentiment_adjustment_pct": round(self.sentiment_pct, 4),
            "whale_adjustment_pct": round(self.whale_pct, 4),
            "total_adjustment_pct": round(self.total * 100.0, 4),
        }


def compute_adjustment(
    sentiment_z: float | None,
    netflow_z: float | None,
    weights: SignalWeights,
    *,
    scale: float = ADJUSTMENT_SCALE,
) -> Adjustment:
    sent = clip_z(sentiment_z)
    whale = clip_z(NETFLOW_SIGN * netflow_z if netflow_z is not None else None)
    sentiment_part = weights.sentiment * sent * scale
    whale_part = weights.whale * whale * scale
    return Adjustment(
        sentiment_z=sent,
        whale_z=whale,
        sentiment_pct=sentiment_part * 100.0,
        whale_pct=whale_part * 100.0,
        total=sentiment_part + whale_part,
    )


def apply_adjustment(values: Sequence[float], total: float, total_steps: int) -> list[float]:
    """Multiply point ``k`` (1-based) by ``1 + total * k / total_steps``."""
    n = max(1, total_steps)
    return [float(v) * (1.0 + total * (k / n)) for k, v in enumerate(values, start=1)]


def classify_direction(log_move: float, sigma_h: float) -> tuple[Direction, float]:
    """Direction of the expected move; returns the flat threshold (percent) used for the call."""
    threshold = FLAT_SIGMA_FRACTION * sigma_h
    threshold_pct = (math.exp(threshold) - 1.0) * 100.0
    if abs(log_move) < threshold:
        return Direction.FLAT, threshold_pct
    return (Direction.UP if log_move > 0 else Direction.DOWN), threshold_pct


def confidence_score(*, relative_width: float, hits: int, evaluated: int, samples: int) -> float:
    """0..1 score from band tightness, rolling directional accuracy and data sufficiency.

    * interval score ``1 / (1 + 10 * (upper - lower) / target)``
    * accuracy score ``(hits + 0.5 * prior) / (evaluated + prior)`` (Laplace-style prior of 50%)
    * data score ``min(1, samples / 720)``
    """
    interval_score = 1.0 / (1.0 + INTERVAL_SHARPNESS * max(0.0, relative_width))
    accuracy_score = (hits + 0.5 * ACCURACY_PRIOR_WEIGHT) / (evaluated + ACCURACY_PRIOR_WEIGHT)
    data_score = min(1.0, max(0.0, samples / FULL_DATA_CANDLES))
    w_int, w_acc, w_data = CONFIDENCE_WEIGHTS
    score = w_int * interval_score + w_acc * accuracy_score + w_data * data_score
    low, high = CONFIDENCE_BOUNDS
    return round(min(high, max(low, score)), 4)


def risk_reward(price: float, target: float, atr_value: float | None, steps: int) -> tuple[float | None, float | None]:
    """``|target - price| / stop`` with ``stop = 1.5 * ATR(14) * sqrt(steps)``; None without a usable ATR."""
    if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
        return None, None
    stop = STOP_ATR_MULTIPLIER * atr_value * math.sqrt(max(1, steps))
    return round(abs(target - price) / stop, 4), round(stop, 8)


@dataclass(slots=True)
class HorizonForecast:
    horizon: str
    steps: int
    target_price: float
    lower: float
    upper: float
    change_pct: float
    direction: Direction
    confidence: float
    risk_reward: float | None
    stop_distance: float | None
    flat_threshold_pct: float
    target_time: datetime
    model: str
    technical_target: float
    adjustment_pct: float
    factors: dict[str, float] = field(default_factory=dict)
    path_points: list[float] = field(default_factory=list)
    path_lower: list[float] = field(default_factory=list)
    path_upper: list[float] = field(default_factory=list)

    @property
    def expected_log_move(self) -> float:
        return math.log(self.target_price / (self.target_price / (1.0 + self.change_pct / 100.0)))


def assemble_horizons(
    *,
    horizons: Sequence[str],
    forecast: ForecastResult,
    price: float,
    origin: datetime,
    base_interval: str = "1h",
    atr_value: float | None,
    sentiment_z: float | None,
    netflow_z: float | None,
    weights: SignalWeights,
    accuracy: Mapping[str, tuple[int, int]] | None = None,
    samples: int,
) -> list[HorizonForecast]:
    """Turn a multi-step technical forecast into one adjusted, scored forecast per horizon.

    ``origin`` is the open time of the candle the forecast starts from; step ``k`` targets
    ``origin + k * base_interval`` (the close of the k-th candle ahead). The full hourly path
    (with band) is attached to the longest horizon for charting.
    """
    accuracy = accuracy or {}
    adjustment = compute_adjustment(sentiment_z, netflow_z, weights)
    total_steps = len(forecast.points)
    points = apply_adjustment(forecast.points, adjustment.total, total_steps)
    lower = apply_adjustment(forecast.lower, adjustment.total, total_steps)
    upper = apply_adjustment(forecast.upper, adjustment.total, total_steps)
    step_delta = timedelta(seconds=interval_seconds(base_interval))
    longest = max((horizon_steps(h, base_interval) for h in horizons), default=total_steps)

    out: list[HorizonForecast] = []
    for horizon in horizons:
        steps = min(horizon_steps(horizon, base_interval), total_steps)
        target, lo, hi = points[steps - 1], lower[steps - 1], upper[steps - 1]
        technical = forecast.points[steps - 1]
        log_move = math.log(target / price) if price > 0 and target > 0 else 0.0
        sigma_h = forecast.residual_std * math.sqrt(steps)
        direction, flat_pct = classify_direction(log_move, sigma_h)
        hits, evaluated = accuracy.get(horizon, (0, 0))
        confidence = confidence_score(
            relative_width=(hi - lo) / target if target > 0 else 1.0, hits=hits, evaluated=evaluated, samples=samples
        )
        rr, stop = risk_reward(price, target, atr_value, steps)
        attach_path = steps == longest
        out.append(
            HorizonForecast(
                horizon=horizon,
                steps=steps,
                target_price=round(target, 8),
                lower=round(lo, 8),
                upper=round(hi, 8),
                change_pct=round((target / price - 1.0) * 100.0, 4) if price > 0 else 0.0,
                direction=direction,
                confidence=confidence,
                risk_reward=rr,
                stop_distance=stop,
                flat_threshold_pct=round(flat_pct, 4),
                target_time=origin + step_delta * steps,
                model=forecast.model,
                technical_target=round(technical, 8),
                adjustment_pct=round(adjustment.total * 100.0 * steps / max(1, total_steps), 4),
                factors=adjustment.as_factors(),
                path_points=[round(v, 8) for v in points] if attach_path else [],
                path_lower=[round(v, 8) for v in lower] if attach_path else [],
                path_upper=[round(v, 8) for v in upper] if attach_path else [],
            )
        )
    return out


__all__ = [
    "ADJUSTMENT_SCALE",
    "DEFAULT_WEIGHTS",
    "FLAT_SIGMA_FRACTION",
    "FULL_DATA_CANDLES",
    "MIN_CANDLES",
    "NETFLOW_SIGN",
    "STOP_ATR_MULTIPLIER",
    "WEIGHT_TOLERANCE",
    "Adjustment",
    "HorizonForecast",
    "SignalWeights",
    "apply_adjustment",
    "assemble_horizons",
    "classify_direction",
    "clip_z",
    "compute_adjustment",
    "confidence_score",
    "horizon_steps",
    "risk_reward",
]
