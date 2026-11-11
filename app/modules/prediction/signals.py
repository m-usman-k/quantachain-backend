"""Automated trade signals (FE-4).

For every fresh prediction ``z = expected_move / forecast_vol_for_horizon`` where the
expected move is the forecast log return and the volatility is the cumulative sigma over
the same horizon from ``volatility.py``. ``|z| >= threshold`` emits BUY/SELL,
``|z| >= 2 * threshold`` STRONG_BUY/STRONG_SELL. A symbol/horizon never carries two
unexpired signals in the same direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.events import EventBus, Topics, event_bus
from app.core.timeutils import utcnow
from app.modules.notifications.models import NotificationType, Severity
from app.modules.notifications.service import NotificationService
from app.modules.prediction.models import Direction, Prediction, SignalAction, TradeSignal
from app.modules.prediction.repository import TradeSignalRepository

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class SignalDecision:
    action: SignalAction
    direction: Direction
    strength: float  # z-score
    expected_move_pct: float
    sigma_pct: float

    @property
    def is_strong(self) -> bool:
        return self.action in (SignalAction.STRONG_BUY, SignalAction.STRONG_SELL)


def decide(expected_log_move: float, horizon_sigma: float, threshold: float) -> SignalDecision | None:
    """Threshold the move/volatility ratio; None when no signal should fire."""
    if not math.isfinite(expected_log_move) or not math.isfinite(horizon_sigma) or horizon_sigma <= 0:
        return None
    z = expected_log_move / horizon_sigma
    if abs(z) < threshold:
        return None
    strong = abs(z) >= 2.0 * threshold
    if z > 0:
        action = SignalAction.STRONG_BUY if strong else SignalAction.BUY
        direction = Direction.UP
    else:
        action = SignalAction.STRONG_SELL if strong else SignalAction.SELL
        direction = Direction.DOWN
    return SignalDecision(
        action=action,
        direction=direction,
        strength=round(z, 4),
        expected_move_pct=round((math.exp(expected_log_move) - 1.0) * 100.0, 4),
        sigma_pct=round((math.exp(horizon_sigma) - 1.0) * 100.0, 4),
    )


def build_rationale(prediction: Prediction, decision: SignalDecision) -> str:
    snapshot = prediction.feature_snapshot or {}
    parts = [
        f"{prediction.horizon} forecast {decision.expected_move_pct:+.2f}% vs {decision.sigma_pct:.2f}% expected "
        f"volatility (z={decision.strength:+.2f}σ, model={prediction.model}, confidence={prediction.confidence:.2f})"
    ]
    rsi = snapshot.get("rsi_14")
    if rsi is not None:
        parts.append(f"RSI(14) {rsi:.0f}")
    macd = snapshot.get("macd_hist")
    if macd is not None:
        parts.append("MACD histogram " + ("positive" if macd > 0 else "negative"))
    sentiment = snapshot.get("sentiment_index")
    if sentiment is not None:
        parts.append(f"sentiment index {sentiment:.0f}")
    netflow = snapshot.get("netflow_z")
    if netflow is not None:
        parts.append(f"exchange netflow z {netflow:+.2f}")
    return "; ".join(parts)


class SignalEngine:
    def __init__(
        self,
        db: AsyncDatabase[dict[str, Any]],
        settings: Settings | None = None,
        *,
        bus: EventBus | None = None,
        notifications: NotificationService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repo = TradeSignalRepository(db)
        self.bus = bus or event_bus
        self.notifications = notifications or NotificationService(db, self.settings)

    async def emit(
        self, prediction: Prediction, horizon_sigma: float, *, now: datetime | None = None, notify: bool = True
    ) -> TradeSignal | None:
        """Create, publish and notify a signal for ``prediction`` unless one in that direction is still active."""
        now = now or utcnow()
        if prediction.price <= 0 or prediction.target_price <= 0:
            return None
        decision = decide(
            math.log(prediction.target_price / prediction.price), horizon_sigma, self.settings.signal_sigma_threshold
        )
        if decision is None or prediction.target_time <= now:
            return None
        if await self.repo.has_active(prediction.symbol, prediction.horizon, decision.direction, now):
            return None
        stop_distance = prediction.stop_distance or abs(prediction.target_price - prediction.price) / 2.0
        stop_loss = (
            prediction.price - stop_distance if decision.direction == Direction.UP else prediction.price + stop_distance
        )
        signal = await self.repo.insert(
            TradeSignal(
                symbol=prediction.symbol,
                horizon=prediction.horizon,
                action=decision.action,
                direction=decision.direction,
                strength=decision.strength,
                confidence=prediction.confidence,
                entry_price=prediction.price,
                target_price=prediction.target_price,
                stop_loss=round(max(stop_loss, 0.0), 8),
                expected_move_pct=decision.expected_move_pct,
                sigma_pct=decision.sigma_pct,
                sigma_threshold=self.settings.signal_sigma_threshold,
                rationale=build_rationale(prediction, decision),
                expires_at=prediction.target_time,
                prediction_id=prediction.id,
                model=prediction.model,
            )
        )
        payload = signal_payload(signal)
        self.bus.publish(Topics.SIGNAL, payload)
        if notify:
            try:
                await self.notifications.notify_subscribers(
                    type=NotificationType.TRADE_SIGNAL,
                    title=f"{signal.symbol} {signal.horizon} {signal.action.replace('_', ' ')} signal",
                    body=signal.rationale,
                    severity=Severity.WARNING if decision.is_strong else Severity.INFO,
                    data={"signal_id": signal.id, "symbol": signal.symbol, "horizon": signal.horizon},
                )
            except Exception as exc:  # pragma: no cover - notifications must never break the engine
                logger.warning("signal_notification_failed", error=str(exc), signal_id=signal.id)
        logger.info(
            "trade_signal_emitted",
            symbol=signal.symbol,
            horizon=signal.horizon,
            action=str(signal.action),
            z=signal.strength,
        )
        return signal


def signal_payload(signal: TradeSignal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "symbol": signal.symbol,
        "horizon": signal.horizon,
        "action": str(signal.action),
        "direction": str(signal.direction),
        "strength": signal.strength,
        "confidence": signal.confidence,
        "entry_price": signal.entry_price,
        "target_price": signal.target_price,
        "stop_loss": signal.stop_loss,
        "expected_move_pct": signal.expected_move_pct,
        "rationale": signal.rationale,
        "expires_at": signal.expires_at.isoformat(),
        "created_at": signal.created_at.isoformat(),
    }


__all__ = ["SignalDecision", "SignalEngine", "build_rationale", "decide", "signal_payload"]
