from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.research.models import ChatMessage, ChatRole, ChatSession, InsightPeriod, MarketInsight


class ChartResponse(BaseModel):
    symbol: str
    interval: str
    range: str
    from_: datetime = Field(alias="from")
    to: datetime
    series: dict[str, list[dict[str, Any]]]

    model_config = {"populate_by_name": True}


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    context: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    pinned: bool | None = None
    archived: bool | None = None


class SessionOut(BaseModel):
    id: str
    title: str
    model: str
    message_count: int
    last_message_at: datetime | None
    pinned: bool
    archived: bool
    context: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_model(cls, s: ChatSession) -> SessionOut:
        return cls(
            id=s.id or "",
            title=s.title,
            model=s.model,
            message_count=s.message_count,
            last_message_at=s.last_message_at,
            pinned=s.pinned,
            archived=s.archived,
            context=s.context,
            created_at=s.created_at,
        )


class MessageOut(BaseModel):
    id: str
    role: ChatRole
    content: str
    tool_calls: list[dict[str, Any]]
    data: dict[str, Any]
    model: str | None
    tokens: int | None
    latency_ms: float | None
    created_at: datetime

    @classmethod
    def from_model(cls, m: ChatMessage) -> MessageOut:
        return cls(
            id=m.id or "",
            role=m.role,
            content=m.content,
            tool_calls=m.tool_calls,
            data=m.data,
            model=m.model,
            tokens=m.tokens,
            latency_ms=m.latency_ms,
            created_at=m.created_at,
        )


class SessionDetail(SessionOut):
    messages: list[MessageOut]


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ChatReply(BaseModel):
    session: SessionOut
    user_message: MessageOut
    assistant_message: MessageOut


class InsightOut(BaseModel):
    id: str
    period: InsightPeriod
    period_start: datetime
    period_end: datetime
    title: str
    summary: str
    highlights: list[str]
    metrics: dict[str, Any]
    performers: dict[str, Any]
    generated_by: str
    in_progress: bool
    updated_at: datetime | None

    @classmethod
    def from_model(cls, i: MarketInsight) -> InsightOut:
        return cls(
            id=i.id or "",
            period=i.period,
            period_start=i.period_start,
            period_end=i.period_end,
            title=i.title,
            summary=i.summary,
            highlights=i.highlights,
            metrics=i.metrics,
            performers=i.performers,
            generated_by=i.generated_by,
            in_progress=i.in_progress,
            updated_at=i.updated_at,
        )


class SavedReportOut(BaseModel):
    id: str
    title: str
    type: str
    symbol: str | None
    status: str
    format: str
    created_at: datetime
    size_bytes: int | None


class LibraryOut(BaseModel):
    sessions: list[SessionOut]
    saved_reports: list[SavedReportOut]
    trending_topics: list[dict[str, Any]]
    latest_insight: InsightOut | None
    assistant_mode: str


class ResearchPulse(BaseModel):
    assistant_mode: str
    messages_last_hour: int
    active_sessions_24h: int
    tool_calls_last_hour: int
    avg_latency_ms: float | None
    llm_tokens_total: int
    generated_at: datetime


__all__ = [name for name in globals() if name[0].isupper()]
