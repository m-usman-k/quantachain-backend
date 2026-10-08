"""Module 8 - Interactive Research Portal endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import DB, AdminUser, CurrentUser, SettingsDep
from app.core.rate_limit import rate_limited
from app.modules.research.charts import CHART_METRICS
from app.modules.research.models import InsightPeriod
from app.modules.research.schemas import (
    ChartResponse,
    ChatReply,
    InsightOut,
    LibraryOut,
    MessageCreate,
    ResearchPulse,
    SessionCreate,
    SessionDetail,
    SessionOut,
    SessionUpdate,
)
from app.modules.research.service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])


def get_research_service(db: DB, settings: SettingsDep) -> ResearchService:
    return ResearchService(db, settings)


Service = Annotated[ResearchService, Depends(get_research_service)]
chat_limiter = rate_limited(30)


@router.get(
    "/charts/{symbol}",
    response_model=ChartResponse,
    response_model_by_alias=True,
    summary="Aligned price / sentiment / whale / forecast series",
)
async def chart(
    symbol: str,
    service: Service,
    metrics: Annotated[
        str, Query(description=f"Comma separated subset of {list(CHART_METRICS)}")
    ] = "price,volume,sentiment",
    interval: Annotated[str, Query(pattern="^(1m|5m|15m|1h|4h|1d)$")] = "1h",
    range: Annotated[str, Query(pattern=r"^\d+[hdw]$")] = "30d",
) -> ChartResponse:
    wanted = [m.strip() for m in metrics.split(",") if m.strip()]
    data = await service.chart(symbol, metrics=wanted, interval=interval, range_=range)
    return ChartResponse(**data)


@router.get(
    "/library",
    response_model=LibraryOut,
    summary="Research library sidebar: sessions, saved reports, trending, latest insight",
)
async def library(user: CurrentUser, service: Service) -> LibraryOut:
    return await service.library(user)


@router.get("/pulse", response_model=ResearchPulse, summary="Research Pulse: assistant load and mode")
async def pulse(_: CurrentUser, service: Service) -> ResearchPulse:
    return await service.pulse()


# -------------------------------------------------------------------- chat
@router.post("/chat/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(payload: SessionCreate, user: CurrentUser, service: Service) -> SessionOut:
    return SessionOut.from_model(await service.create_session(user, title=payload.title, context=payload.context))


@router.get("/chat/sessions", response_model=list[SessionOut])
async def list_sessions(
    user: CurrentUser, service: Service, include_archived: Annotated[bool, Query()] = False
) -> list[SessionOut]:
    return [SessionOut.from_model(s) for s in await service.list_sessions(user, include_archived=include_archived)]


@router.get("/chat/sessions/{session_id}", response_model=SessionDetail)
async def session_detail(session_id: str, user: CurrentUser, service: Service) -> SessionDetail:
    return await service.session_detail(user, session_id)


@router.patch("/chat/sessions/{session_id}", response_model=SessionOut)
async def update_session(session_id: str, payload: SessionUpdate, user: CurrentUser, service: Service) -> SessionOut:
    return SessionOut.from_model(
        await service.update_session(
            user, session_id, title=payload.title, pinned=payload.pinned, archived=payload.archived
        )
    )


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, user: CurrentUser, service: Service) -> None:
    await service.delete_session(user, session_id)


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatReply,
    dependencies=[Depends(chat_limiter)],
    summary="Ask the research assistant (multi-turn)",
)
async def send_message(session_id: str, payload: MessageCreate, user: CurrentUser, service: Service) -> ChatReply:
    return await service.send_message(user, session_id, payload.content)


@router.post(
    "/chat/quick",
    response_model=ChatReply,
    dependencies=[Depends(chat_limiter)],
    summary="One-shot question (creates a session)",
)
async def quick_ask(payload: MessageCreate, user: CurrentUser, service: Service) -> ChatReply:
    return await service.quick_ask(user, payload.content)


# ---------------------------------------------------------------- insights
@router.get("/insights", response_model=list[InsightOut], summary="Monthly / quarterly macro insights")
async def insights(
    service: Service,
    period: Annotated[InsightPeriod | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=24)] = 6,
) -> list[InsightOut]:
    return await service.list_insights(period, limit=limit)


@router.post("/insights/refresh", response_model=list[InsightOut], summary="Recompute insights now (admin)")
async def refresh_insights(_: AdminUser, service: Service) -> list[InsightOut]:
    return await service.refresh_insights()
