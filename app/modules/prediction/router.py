"""Module 5 - AI Prediction & Forecasting endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse

from app.api.deps import DB, AdminUser, CurrentUser, SettingsDep
from app.core.pagination import Page, Pagination
from app.core.rate_limit import rate_limited
from app.modules.prediction.schemas import (
    AccuracyOut,
    HorizonOut,
    MatrixOut,
    MLModelOut,
    ModelsStatusOut,
    RefreshResult,
    ReportOut,
    SignalOut,
    SymbolForecastOut,
    TrainRequest,
    VolatilityOut,
    WeightsIn,
    WeightsOut,
)
from app.modules.prediction.service import PredictionService

router = APIRouter(prefix="/predictions", tags=["prediction"])


def get_prediction_service(db: DB, settings: SettingsDep) -> PredictionService:
    return PredictionService(db, settings)


Service = Annotated[PredictionService, Depends(get_prediction_service)]


# Static routes are declared before ``/{symbol}`` so they are matched first.
@router.get("", response_model=MatrixOut, summary="Cross-timeframe matrix for every tracked asset")
async def matrix(service: Service) -> MatrixOut:
    return await service.matrix()


@router.get(
    "/accuracy", response_model=AccuracyOut, summary="Measured directional accuracy / MAPE of evaluated forecasts"
)
async def accuracy(
    service: Service,
    symbol: Annotated[str | None, Query()] = None,
    window_days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> AccuracyOut:
    return await service.accuracy(symbol=symbol.upper() if symbol else None, window_days=window_days)


@router.get("/signals", response_model=list[SignalOut], summary="Automated trade signals")
async def signals(
    service: Service,
    symbol: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    active_only: Annotated[bool, Query()] = False,
) -> list[SignalOut]:
    return await service.list_signals(symbol=symbol.upper() if symbol else None, limit=limit, active_only=active_only)


@router.get("/weights", response_model=WeightsOut, summary="Model signal weights (technical / whale / sentiment)")
async def weights(service: Service) -> WeightsOut:
    return await service.get_weights()


@router.put("/weights", response_model=WeightsOut, summary="Update signal weights (admin; must sum to 1)")
async def update_weights(payload: WeightsIn, admin: AdminUser, service: Service) -> WeightsOut:
    return await service.set_weights(payload.technical, payload.whale, payload.sentiment, actor=admin.email)


@router.get("/models", response_model=ModelsStatusOut, summary="Forecaster status and trained LSTM models (admin)")
async def models(_: AdminUser, service: Service) -> ModelsStatusOut:
    return await service.models_status()


@router.post(
    "/models/{symbol}/train",
    response_model=MLModelOut,
    summary="Train the LSTM for a symbol (admin; requires TensorFlow)",
)
async def train(symbol: str, _: AdminUser, service: Service, payload: TrainRequest | None = None) -> MLModelOut:
    return await service.train_lstm(symbol, epochs=(payload or TrainRequest()).epochs)


@router.post("/refresh", response_model=RefreshResult, summary="Run a forecast refresh now (admin)")
async def refresh(_: AdminUser, service: Service, symbol: Annotated[str | None, Query()] = None) -> RefreshResult:
    return await service.refresh_all(symbols=[symbol.upper()] if symbol else None)


@router.post(
    "/evaluate", response_model=dict[str, int], summary="Evaluate matured forecasts against realised prices (admin)"
)
async def evaluate(_: AdminUser, service: Service) -> dict[str, int]:
    return await service.evaluate_pending()


@router.get("/reports", response_model=Page[ReportOut], summary="My generated reports")
async def reports(user: CurrentUser, service: Service, settings: SettingsDep, page: Pagination) -> Page[ReportOut]:
    items, total = await service.list_reports(user, page)
    return Page.build([ReportOut.from_model(r, prefix=settings.api_v1_prefix) for r in items], total, page)


@router.get("/reports/{report_id}/download", summary="Download a generated report")
async def download_report(report_id: str, user: CurrentUser, service: Service) -> FileResponse:
    report = await service.report_for_download(user, report_id)
    filename = f"{report.symbol or 'report'}-{report.type}-{report.created_at.strftime('%Y%m%d')}.{report.format}"
    return FileResponse(
        report.file_path or "",
        media_type="application/pdf" if report.format == "pdf" else "text/csv",
        filename=filename,
    )


@router.post(
    "/{symbol}/report",
    response_model=ReportOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limited(10))],
    summary="Generate a confidence-scored PDF forecast report",
)
async def create_report(symbol: str, user: CurrentUser, service: Service, settings: SettingsDep) -> ReportOut:
    return ReportOut.from_model(await service.generate_report(user, symbol), prefix=settings.api_v1_prefix)


@router.get(
    "/{symbol}", response_model=SymbolForecastOut, summary="Latest forecasts, path, volatility, signals and model info"
)
async def symbol_forecast(symbol: str, service: Service) -> SymbolForecastOut:
    return await service.symbol_forecast(symbol)


@router.get("/{symbol}/history", response_model=list[HorizonOut], summary="Past forecasts with realised outcomes")
async def history(
    symbol: str,
    service: Service,
    horizon: Annotated[str, Query()] = "24h",
    range: Annotated[str, Query(pattern=r"^\d+[hdw]$")] = "7d",
) -> list[HorizonOut]:
    return await service.history(symbol.upper(), horizon=horizon, range_=range)


@router.get(
    "/{symbol}/volatility", response_model=VolatilityOut, summary="Volatility estimators, 24h forecast and regime"
)
async def volatility(symbol: str, service: Service) -> VolatilityOut:
    return await service.volatility_for(symbol)
