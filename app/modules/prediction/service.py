"""Module 5 - AI Prediction & Forecasting Engine service."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.events import EventBus, Topics, event_bus
from app.core.exceptions import FeatureDisabledError, NotFoundError, ServiceUnavailableError, ValidationFailedError
from app.core.pagination import PageParams
from app.core.timeutils import parse_range, utcnow
from app.db.collections import Collections
from app.modules.admin.controls import ModelControlService
from app.modules.auth.models import Role, User
from app.modules.market.models import Candle
from app.modules.market.repository import AssetRepository, AssetStatsRepository, CandleRepository
from app.modules.prediction.ensemble import MIN_CANDLES, SignalWeights, assemble_horizons, horizon_steps
from app.modules.prediction.features import FeatureFrame, build_features, candle_arrays
from app.modules.prediction.forecasters import Forecaster, LSTMForecaster, StatisticalForecaster
from app.modules.prediction.models import (
    Direction,
    MLModel,
    Prediction,
    Report,
    ReportFormat,
    ReportStatus,
    ReportType,
    VolatilityForecast,
)
from app.modules.prediction.reporting import build_token_report
from app.modules.prediction.repository import (
    WEIGHTS_STATE_KEY,
    MLModelRepository,
    PredictionRepository,
    PredictionStateStore,
    ReportRepository,
    TradeSignalRepository,
    VolatilityRepository,
)
from app.modules.prediction.schemas import (
    AccuracyBucket,
    AccuracyOut,
    HorizonOut,
    MatrixOut,
    MatrixRow,
    MLModelOut,
    ModelInfo,
    ModelsStatusOut,
    PathPoint,
    RefreshResult,
    SignalOut,
    SymbolForecastOut,
    VolatilityOut,
    WeightsOut,
)
from app.modules.prediction.signals import SignalEngine
from app.modules.prediction.volatility import VolatilitySnapshot, forecast_volatility

logger = structlog.get_logger(__name__)

HISTORY_CANDLES = 1000
DAILY_CANDLES = 400
EXOGENOUS_DAYS = 30
EVALUATION_GRACE = timedelta(hours=2)


class PredictionService:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, *, bus: EventBus | None = None
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self.predictions = PredictionRepository(db)
        self.volatility = VolatilityRepository(db)
        self.signals = TradeSignalRepository(db)
        self.ml_models = MLModelRepository(db)
        self.reports = ReportRepository(db)
        self.state = PredictionStateStore(db)
        self.candles = CandleRepository(db)
        self.assets = AssetRepository(db)
        self.stats = AssetStatsRepository(db)
        self.controls = ModelControlService(db)
        self.signal_engine = SignalEngine(db, self.settings, bus=self.bus)
        self.model_dir = Path(self.settings.data_dir) / "models"
        self.report_dir = Path(self.settings.data_dir) / "reports"

    # ------------------------------------------------------------ weights
    async def get_weights(self) -> WeightsOut:
        stored = await self.state.get(WEIGHTS_STATE_KEY)
        if stored:
            try:
                weights = SignalWeights(**{k: stored[k] for k in ("technical", "whale", "sentiment")})
                return WeightsOut(
                    **weights.as_dict(), updated_at=stored.get("updated_at"), updated_by=stored.get("updated_by")
                )
            except (KeyError, ValueError):
                logger.warning("invalid_stored_weights_using_defaults")
        return WeightsOut(**SignalWeights().as_dict())

    async def set_weights(self, technical: float, whale: float, sentiment: float, *, actor: str | None) -> WeightsOut:
        try:
            weights = SignalWeights(technical=technical, whale=whale, sentiment=sentiment)
        except ValueError as exc:
            raise ValidationFailedError(str(exc)) from exc
        await self.state.set(
            WEIGHTS_STATE_KEY, {**weights.as_dict(), "updated_at": utcnow(), "updated_by": actor}, updated_by=actor
        )
        return await self.get_weights()

    async def _signal_weights(self) -> SignalWeights:
        current = await self.get_weights()
        return SignalWeights(technical=current.technical, whale=current.whale, sentiment=current.sentiment)

    # ------------------------------------------------------------- inputs
    async def _exogenous(self, symbol: str) -> tuple[list[tuple[datetime, float]], list[tuple[datetime, float]]]:
        since = utcnow() - timedelta(days=EXOGENOUS_DAYS)
        sentiment_rows = (
            await self.db[Collections.SENTIMENT_SNAPSHOTS]
            .find({"meta.scope": "asset", "meta.symbol": symbol, "ts": {"$gte": since}}, {"ts": 1, "index": 1})
            .sort("ts", 1)
            .to_list(length=5000)
        )
        sentiment = [(r["ts"], float(r["index"])) for r in sentiment_rows if r.get("index") is not None]
        netflow_rows = (
            await self.db[Collections.ONCHAIN_METRICS]
            .find(
                {"meta.metric": "exchange_netflow_usd", "meta.symbol": symbol, "ts": {"$gte": since}},
                {"ts": 1, "value": 1},
            )
            .sort("ts", 1)
            .to_list(length=5000)
        )
        netflow = [(r["ts"], float(r["value"])) for r in netflow_rows if r.get("value") is not None]
        return sentiment, netflow

    async def _load(self, symbol: str) -> tuple[list[Candle], list[Candle], FeatureFrame]:
        hourly = await self.candles.range(symbol, "1h", limit=HISTORY_CANDLES)
        daily = await self.candles.range(symbol, "1d", limit=DAILY_CANDLES)
        sentiment, netflow = await self._exogenous(symbol)
        frame = build_features(hourly, sentiment=sentiment or None, netflow=netflow or None)
        return hourly, daily, frame

    def _select_forecaster(self, symbol: str) -> Forecaster:
        if self.settings.lstm_enabled and LSTMForecaster.available():
            lstm = LSTMForecaster(self.model_dir, symbol=symbol, interval="1h", lookback=self.settings.lstm_lookback)
            if lstm.has_model():
                return lstm
        return StatisticalForecaster()

    # ------------------------------------------------------------ forecast
    async def forecast_symbol(self, symbol: str, *, emit_signals: bool = True) -> dict[str, Any]:
        symbol = symbol.upper()
        hourly, daily, frame = await self._load(symbol)
        if len(hourly) < MIN_CANDLES:
            raise ValidationFailedError(f"{symbol} has only {len(hourly)} hourly candles; need {MIN_CANDLES}")
        arrays = candle_arrays(hourly)
        closes: np.ndarray = arrays["close"]
        price = float(closes[-1])
        horizons = list(self.settings.prediction_horizons)
        steps = max(horizon_steps(h) for h in horizons)

        forecaster = self._select_forecaster(symbol)
        try:
            if isinstance(forecaster, LSTMForecaster):
                forecaster.fit(closes, frame.matrix(list(forecaster.features), fill=0.0))
            else:
                forecaster.fit(closes)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.warning("lstm_unusable_falling_back", symbol=symbol, error=str(exc))
            forecaster = StatisticalForecaster().fit(closes)
        result = forecaster.predict(steps)

        daily_arrays = candle_arrays(daily) if daily else None
        vol: VolatilitySnapshot = forecast_volatility(
            high=arrays["high"],
            low=arrays["low"],
            close=closes,
            times=arrays["times"],
            daily_closes=daily_arrays["close"] if daily_arrays is not None else None,
            daily_times=daily_arrays["times"] if daily_arrays is not None else None,
        )
        accuracy = {h: await self.predictions.recent_hits(symbol, h) for h in horizons}
        weights = await self._signal_weights()

        def nan_to_none(value: float) -> float | None:
            return None if math.isnan(value) else value

        assembled = assemble_horizons(
            horizons=horizons,
            forecast=result,
            price=price,
            origin=arrays["times"][-1],
            atr_value=nan_to_none(frame.last_value("atr_14", math.nan)),
            sentiment_z=nan_to_none(frame.last_value("sentiment_z", math.nan)),
            netflow_z=nan_to_none(frame.last_value("netflow_z", math.nan)),
            weights=weights,
            accuracy=accuracy,
            samples=len(hourly),
        )
        snapshot = frame.latest()
        stored: list[Prediction] = []
        for h in assembled:
            prediction = Prediction(
                symbol=symbol,
                horizon=h.horizon,
                steps=h.steps,
                price=price,
                target_price=h.target_price,
                lower=h.lower,
                upper=h.upper,
                change_pct=h.change_pct,
                direction=h.direction,
                confidence=h.confidence,
                risk_reward=h.risk_reward,
                stop_distance=h.stop_distance,
                flat_threshold_pct=h.flat_threshold_pct,
                target_time=h.target_time,
                model=h.model,
                technical_target=h.technical_target,
                adjustment_pct=h.adjustment_pct,
                factors=h.factors,
                weights=weights.as_dict(),
                feature_snapshot=snapshot,
                path_points=h.path_points,
                path_lower=h.path_lower,
                path_upper=h.path_upper,
                samples=len(hourly),
            )
            stored.append(await self.predictions.store_latest(prediction))

        vol_doc = await self.volatility.store_latest(
            VolatilityForecast(
                symbol=symbol,
                realized_24h_pct=round(vol.realized_24h * 100, 4),
                realized_7d_pct=round(vol.realized_7d * 100, 4),
                parkinson_24h_pct=round(vol.parkinson_24h * 100, 4),
                ewma_pct=round(vol.ewma * 100, 4),
                garch_pct=round(vol.garch * 100, 4) if vol.garch is not None else None,
                forecast_24h_annualised_pct=round(vol.forecast_24h_annualised * 100, 4),
                forecast_24h_horizon_pct=round(vol.forecast_24h_horizon * 100, 4),
                horizon_sigmas=vol.horizon_sigmas,
                regime=vol.regime,
                percentile=vol.percentile,
                observations=vol.observations,
                history_times=vol.history_times,
                history_values_pct=[round(v * 100, 4) for v in vol.history_values],
                params={k: float(v) for k, v in vol.params.items()},
            )
        )

        emitted = 0
        if emit_signals and not await self.controls.is_paused("signal_engine"):
            for prediction in stored:
                signal = await self.signal_engine.emit(prediction, vol.horizon_sigma(prediction.steps))
                emitted += 1 if signal else 0

        self.bus.publish(
            Topics.PREDICTION,
            {
                "symbol": symbol,
                "price": price,
                "model": result.model,
                "horizons": [
                    {
                        "horizon": p.horizon,
                        "target_price": p.target_price,
                        "change_pct": p.change_pct,
                        "confidence": p.confidence,
                        "direction": str(p.direction),
                    }
                    for p in stored
                ],
            },
        )
        return {
            "symbol": symbol,
            "model": result.model,
            "price": price,
            "predictions": len(stored),
            "signals": emitted,
            "volatility_regime": str(vol_doc.regime),
        }

    async def refresh_all(self, *, symbols: list[str] | None = None) -> RefreshResult:
        started = time.perf_counter()
        if await self.controls.is_paused("price_forecaster"):
            raise ServiceUnavailableError("The price forecaster is paused by an administrator")
        tracked = symbols or [a.symbol for a in await self.assets.tracked()] or list(self.settings.tracked_symbols)
        done: list[str] = []
        skipped: dict[str, str] = {}
        predictions = signals = 0
        for symbol in tracked:
            try:
                summary = await self.forecast_symbol(symbol)
            except ValidationFailedError as exc:
                skipped[symbol] = exc.message
                continue
            except Exception as exc:  # one bad symbol must not stop the batch
                logger.exception("forecast_failed", symbol=symbol)
                skipped[symbol] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            done.append(symbol)
            predictions += summary["predictions"]
            signals += summary["signals"]
        duration = round((time.perf_counter() - started) * 1000, 1)
        await self.controls.record_run(
            "price_forecaster", status="ok" if done else "skipped", symbols=len(done), skipped=len(skipped)
        )
        return RefreshResult(
            symbols=done, skipped=skipped, predictions=predictions, signals=signals, duration_ms=duration
        )

    # ---------------------------------------------------------- evaluation
    async def evaluate_pending(self, *, now: datetime | None = None) -> dict[str, int]:
        now = now or utcnow()
        evaluated = waiting = 0
        for prediction in await self.predictions.pending_evaluation(now):
            candle = await self.candles.col.find_one(
                {"symbol": prediction.symbol, "interval": "1h", "open_time": {"$lte": prediction.target_time}},
                sort=[("open_time", -1)],
            )
            if candle is None or candle["open_time"] < prediction.target_time - EVALUATION_GRACE:
                waiting += 1
                continue
            realized = float(candle["close"])
            move_pct = (realized / prediction.price - 1.0) * 100.0 if prediction.price else 0.0
            if prediction.direction == Direction.FLAT:
                hit = abs(move_pct) <= max(prediction.flat_threshold_pct, 0.05)
            else:
                hit = (move_pct > 0) == (prediction.direction == Direction.UP) and abs(move_pct) > 0
            await self.predictions.update(
                prediction.id or "",
                {
                    "evaluated": True,
                    "realized_price": realized,
                    "abs_error_pct": round(abs(realized - prediction.target_price) / realized * 100.0, 4)
                    if realized
                    else None,
                    "direction_hit": bool(hit),
                    "evaluated_at": now,
                },
            )
            evaluated += 1
        return {"evaluated": evaluated, "waiting_for_data": waiting}

    async def accuracy(self, *, symbol: str | None = None, window_days: int = 30) -> AccuracyOut:
        since = utcnow() - timedelta(days=window_days)
        buckets = await self.predictions.accuracy(symbol=symbol, since=since)
        by_horizon = [
            AccuracyBucket(
                horizon=h,
                count=int(b["count"]),
                hits=int(b["hits"]),
                directional_accuracy_pct=round(100 * b["hits"] / b["count"], 2) if b["count"] else 0.0,
                mape_pct=round(float(b["mape"]), 4),
                avg_confidence=round(float(b["avg_confidence"]), 4),
            )
            for h, b in sorted(buckets.items(), key=lambda kv: horizon_steps(kv[0]) if kv[0][-1] in "hd" else 0)
        ]
        total = sum(b.count for b in by_horizon)
        overall = None
        if total:
            hits = sum(b.hits for b in by_horizon)
            overall = AccuracyBucket(
                horizon="all",
                count=total,
                hits=hits,
                directional_accuracy_pct=round(100 * hits / total, 2),
                mape_pct=round(sum(b.mape_pct * b.count for b in by_horizon) / total, 4),
                avg_confidence=round(sum(b.avg_confidence * b.count for b in by_horizon) / total, 4),
            )
        return AccuracyOut(overall=overall, by_horizon=by_horizon, symbol=symbol, window_days=window_days)

    # ---------------------------------------------------------------- reads
    def _pair(self, symbol: str) -> str:
        return f"{symbol}/{self.settings.quote_asset}"

    async def matrix(self) -> MatrixOut:
        symbols = [a.symbol for a in await self.assets.tracked()] or list(self.settings.tracked_symbols)
        latest = await self.predictions.latest_matrix(symbols)
        stats = {s.symbol: s for s in await self.stats.find({"symbol": {"$in": symbols}})}
        rows = []
        for symbol in symbols:
            horizons = latest.get(symbol, {})
            if not horizons:
                continue
            ordered = sorted(horizons.values(), key=lambda p: p.steps)
            stat = stats.get(symbol)
            rows.append(
                MatrixRow(
                    symbol=symbol,
                    pair=self._pair(symbol),
                    price=stat.price if stat and stat.price is not None else ordered[0].price,
                    horizons=[HorizonOut.from_model(p) for p in ordered],
                    updated_at=ordered[0].created_at,
                )
            )
        accuracy = await self.accuracy()
        return MatrixOut(
            rows=rows,
            horizons=list(self.settings.prediction_horizons),
            aggregate_accuracy_pct=accuracy.overall.directional_accuracy_pct if accuracy.overall else None,
            generated_at=utcnow(),
        )

    async def model_info(self, symbol: str) -> ModelInfo:
        trained = await self.ml_models.active_model("lstm", symbol, "1h")
        available = LSTMForecaster.available()
        in_use = (
            "lstm"
            if (
                available
                and self.settings.lstm_enabled
                and trained is not None
                and LSTMForecaster(self.model_dir, symbol=symbol).has_model()
            )
            else "statistical"
        )
        return ModelInfo(
            forecaster=in_use,
            lstm_available=available,
            lstm_enabled=self.settings.lstm_enabled,
            lstm_trained=trained is not None,
            lstm_metrics=trained.metrics if trained else None,
            trained_at=trained.trained_at if trained else None,
        )

    async def symbol_forecast(self, symbol: str) -> SymbolForecastOut:
        symbol = symbol.upper()
        latest = await self.predictions.latest_for(symbol)
        if not latest:
            if await self.assets.get_by_symbol(symbol) is None:
                raise NotFoundError(f"Asset {symbol} is not tracked")
            raise NotFoundError(
                f"No forecast yet for {symbol}; the engine refreshes every {self.settings.prediction_refresh_seconds // 60} minutes"
            )
        now = utcnow()
        with_path = next((p for p in latest if p.path_points), None)
        path: list[PathPoint] = []
        if with_path:
            origin = with_path.target_time - timedelta(hours=with_path.steps)
            for k, (pt, lo, hi) in enumerate(
                zip(with_path.path_points, with_path.path_lower, with_path.path_upper, strict=False), start=1
            ):
                path.append(PathPoint(t=origin + timedelta(hours=k), point=pt, lower=lo, upper=hi))
        vol = await self.volatility.latest(symbol)
        stat = await self.stats.get_by_symbol(symbol)
        weights = await self.get_weights()
        return SymbolForecastOut(
            symbol=symbol,
            pair=self._pair(symbol),
            price=stat.price if stat and stat.price is not None else latest[0].price,
            horizons=[HorizonOut.from_model(p) for p in latest],
            path=path,
            weights={"technical": weights.technical, "whale": weights.whale, "sentiment": weights.sentiment},
            volatility=VolatilityOut.from_model(vol) if vol else None,
            signals=[SignalOut.from_model(s, now=now) for s in await self.signals.active(symbol, now)],
            model=await self.model_info(symbol),
            accuracy=await self.accuracy(symbol=symbol),
            feature_snapshot=latest[0].feature_snapshot,
            generated_at=now,
        )

    async def history(self, symbol: str, *, horizon: str, range_: str) -> list[HorizonOut]:
        if horizon not in self.settings.prediction_horizons:
            raise ValidationFailedError(
                f"Unknown horizon {horizon}; use one of {list(self.settings.prediction_horizons)}"
            )
        rows = await self.predictions.history(symbol, horizon, since=utcnow() - parse_range(range_))
        return [HorizonOut.from_model(p) for p in rows]

    async def volatility_for(self, symbol: str) -> VolatilityOut:
        vol = await self.volatility.latest(symbol)
        if vol is None:
            raise NotFoundError(f"No volatility forecast yet for {symbol.upper()}")
        return VolatilityOut.from_model(vol)

    async def list_signals(self, *, symbol: str | None, limit: int, active_only: bool) -> list[SignalOut]:
        now = utcnow()
        rows = (
            await self.signals.active(symbol, now, limit=limit)
            if active_only
            else await self.signals.recent(symbol, limit=limit)
        )
        return [SignalOut.from_model(s, now=now) for s in rows]

    # --------------------------------------------------------------- models
    async def models_status(self) -> ModelsStatusOut:
        models = await self.ml_models.list_models()
        available = LSTMForecaster.available()
        return ModelsStatusOut(
            tensorflow_available=available,
            lstm_enabled=self.settings.lstm_enabled,
            forecaster_in_use="lstm"
            if (available and self.settings.lstm_enabled and any(m.active for m in models))
            else "statistical",
            lookback=self.settings.lstm_lookback,
            horizons=list(self.settings.prediction_horizons),
            models=[MLModelOut.from_model(m) for m in models],
            paused={
                name: await self.controls.is_paused(name)
                for name in ("price_forecaster", "volatility", "signal_engine")
            },
        )

    async def train_lstm(self, symbol: str, *, epochs: int = 20) -> MLModelOut:
        if not LSTMForecaster.available():
            raise FeatureDisabledError("TensorFlow is not installed; install requirements-ml.txt to train the LSTM")
        symbol = symbol.upper()
        hourly, _daily, frame = await self._load(symbol)
        if len(hourly) < MIN_CANDLES:
            raise ValidationFailedError(f"{symbol} has only {len(hourly)} hourly candles; need {MIN_CANDLES}")
        forecaster = LSTMForecaster(self.model_dir, symbol=symbol, interval="1h", lookback=self.settings.lstm_lookback)
        closes = candle_arrays(hourly)["close"]
        result = forecaster.train(closes, frame.matrix(list(forecaster.features), fill=0.0), epochs=epochs)
        version = await self.ml_models.next_version("lstm", symbol, "1h")
        model = await self.ml_models.register(
            MLModel(
                name="lstm",
                symbol=symbol,
                interval="1h",
                version=version,
                trained_at=utcnow(),
                samples=result.samples,
                epochs=result.epochs,
                lookback=result.lookback,
                features=result.features,
                metrics=result.metrics,
                scaler=result.scaler,
                path=result.path,
            )
        )
        await self.controls.record_run("price_forecaster", status="trained", symbol=symbol, version=version)
        return MLModelOut.from_model(model)

    async def train_all(self) -> dict[str, Any]:
        trained: list[str] = []
        failed: dict[str, str] = {}
        for asset in await self.assets.tracked():
            try:
                await self.train_lstm(asset.symbol)
                trained.append(asset.symbol)
            except Exception as exc:
                failed[asset.symbol] = f"{type(exc).__name__}: {exc}"[:200]
        return {"trained": trained, "failed": failed}

    # -------------------------------------------------------------- reports
    async def generate_report(self, user: User, symbol: str) -> Report:
        symbol = symbol.upper()
        latest = await self.predictions.latest_for(symbol)
        if not latest:
            await self.forecast_symbol(symbol, emit_signals=False)
            latest = await self.predictions.latest_for(symbol)
        report = await self.reports.insert(
            Report(
                user_id=user.id or "",
                type=ReportType.TOKEN_FORECAST,
                title=f"{symbol} forecast report",
                symbol=symbol,
                format=ReportFormat.PDF,
                params={"horizons": [p.horizon for p in latest]},
            )
        )
        try:
            hourly = await self.candles.range(symbol, "1h", limit=168)
            vol = await self.volatility.latest(symbol)
            sentiment = await self.db[Collections.SENTIMENT_SNAPSHOTS].find_one(
                {"meta.scope": "asset", "meta.symbol": symbol}, sort=[("ts", -1)]
            )
            since = utcnow() - timedelta(hours=24)
            whale_cursor = await self.db[Collections.WHALE_TRANSFERS].aggregate(
                [
                    {"$match": {"symbol": symbol, "block_time": {"$gte": since}}},
                    {
                        "$group": {
                            "_id": None,
                            "transfers": {"$sum": 1},
                            "volume_usd": {"$sum": "$amount_usd"},
                            "inflow": {"$sum": {"$cond": [{"$eq": ["$flow", "exchange_inflow"]}, "$amount_usd", 0]}},
                            "outflow": {"$sum": {"$cond": [{"$eq": ["$flow", "exchange_outflow"]}, "$amount_usd", 0]}},
                        }
                    },
                ]
            )
            whale_rows = await whale_cursor.to_list(length=1)
            whales = (
                {
                    "transfers": whale_rows[0]["transfers"],
                    "volume_usd": whale_rows[0]["volume_usd"],
                    "exchange_netflow_usd": whale_rows[0]["inflow"] - whale_rows[0]["outflow"],
                }
                if whale_rows
                else None
            )
            accuracy = await self.accuracy(symbol=symbol)
            weights = await self.get_weights()
            data = {
                "price": latest[0].price,
                "model": latest[0].model,
                "weights": {"technical": weights.technical, "whale": weights.whale, "sentiment": weights.sentiment},
                "price_history": [(c.open_time, c.close) for c in hourly],
                "horizons": [p.model_dump() for p in latest],
                "volatility": vol.model_dump() if vol else None,
                "sentiment": {
                    "index": sentiment.get("index"),
                    "label": sentiment.get("label"),
                    "sample_size": sentiment.get("sample_size"),
                }
                if sentiment
                else None,
                "whales": whales,
                "accuracy": {
                    "count": accuracy.overall.count,
                    "directional_accuracy_pct": accuracy.overall.directional_accuracy_pct,
                }
                if accuracy.overall
                else None,
            }
            pdf = build_token_report(symbol, data)
            self.report_dir.mkdir(parents=True, exist_ok=True)
            path = self.report_dir / f"{report.id}_{symbol}_forecast.pdf"
            path.write_bytes(pdf)
            return await self.reports.mark_ready(report.id or "", file_path=str(path), size_bytes=len(pdf)) or report
        except Exception as exc:
            logger.exception("report_generation_failed", symbol=symbol)
            return await self.reports.mark_failed(report.id or "", f"{type(exc).__name__}: {exc}") or report

    async def list_reports(self, user: User, params: PageParams) -> tuple[list[Report], int]:
        return await self.reports.for_user(user.id or "", params)

    async def report_for_download(self, user: User, report_id: str) -> Report:
        report = await self.reports.get(report_id)
        if report is None or (report.user_id != user.id and user.role != Role.ADMIN):
            raise NotFoundError("Report not found")
        if report.status != ReportStatus.READY or not report.file_path or not Path(report.file_path).exists():
            raise NotFoundError("Report file is not available")
        return report


__all__ = ["PredictionService"]
