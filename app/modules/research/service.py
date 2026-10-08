"""Module 8 - Interactive Research Portal."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.integrations.llm import LLMClient
from app.modules.auth.models import Role, User
from app.modules.market.repository import AssetRepository
from app.modules.market.symbols import SymbolDetector
from app.modules.research.assistant import ResearchAssistant
from app.modules.research.charts import build_chart
from app.modules.research.insights import compute_insight
from app.modules.research.models import ChatMessage, ChatRole, ChatSession, InsightPeriod
from app.modules.research.repository import ChatMessageRepository, ChatSessionRepository, MarketInsightRepository
from app.modules.research.schemas import (
    ChatReply,
    InsightOut,
    LibraryOut,
    MessageOut,
    ResearchPulse,
    SavedReportOut,
    SessionDetail,
    SessionOut,
)
from app.modules.research.tools import run_tool

_llm_singleton: LLMClient | None = None


def shared_llm(settings: Settings) -> LLMClient:
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMClient(settings)
    return _llm_singleton


class ResearchService:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, *, llm: LLMClient | None = None
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else shared_llm(self.settings)
        self.sessions = ChatSessionRepository(db)
        self.messages = ChatMessageRepository(db)
        self.insights = MarketInsightRepository(db)

    # -------------------------------------------------------------- charts
    async def chart(self, symbol: str, *, metrics: list[str], interval: str, range_: str) -> dict[str, Any]:
        return await build_chart(self.db, symbol, metrics=metrics, interval=interval, range_=range_)

    # ---------------------------------------------------------------- chat
    async def _assistant(self) -> ResearchAssistant:
        detector = SymbolDetector(await AssetRepository(self.db).tracked())
        return ResearchAssistant(self.db, self.settings, llm=self.llm, detector=detector)

    async def create_session(self, user: User, *, title: str | None, context: dict[str, Any]) -> ChatSession:
        assistant = await self._assistant()
        return await self.sessions.insert(
            ChatSession(
                user_id=user.id or "", title=title or "New research session", context=context, model=assistant.mode
            )
        )

    async def list_sessions(self, user: User, *, include_archived: bool = False) -> list[ChatSession]:
        return await self.sessions.for_user(user.id or "", include_archived=include_archived)

    async def get_session(self, user: User, session_id: str) -> ChatSession:
        session = await self.sessions.get(session_id)
        if session is None or (session.user_id != user.id and user.role != Role.ADMIN):
            raise NotFoundError("Chat session not found")
        return session

    async def session_detail(self, user: User, session_id: str) -> SessionDetail:
        session = await self.get_session(user, session_id)
        messages = await self.messages.all_for_session(session_id)
        return SessionDetail(
            **SessionOut.from_model(session).model_dump(), messages=[MessageOut.from_model(m) for m in messages]
        )

    async def update_session(
        self, user: User, session_id: str, *, title: str | None, pinned: bool | None, archived: bool | None
    ) -> ChatSession:
        session = await self.get_session(user, session_id)
        changes = {k: v for k, v in (("title", title), ("pinned", pinned), ("archived", archived)) if v is not None}
        if not changes:
            return session
        return await self.sessions.update(session_id, changes) or session

    async def delete_session(self, user: User, session_id: str) -> None:
        await self.get_session(user, session_id)
        await self.messages.delete_many({"session_id": session_id})
        await self.sessions.delete(session_id)

    async def send_message(self, user: User, session_id: str, content: str) -> ChatReply:
        session = await self.get_session(user, session_id)
        assistant = await self._assistant()
        history_models = await self.messages.history(session_id)
        history = [
            {"role": str(m.role), "content": m.content}
            for m in history_models
            if m.role in (ChatRole.USER, ChatRole.ASSISTANT)
        ]

        user_message = await self.messages.insert(
            ChatMessage(session_id=session_id, user_id=user.id or "", role=ChatRole.USER, content=content)
        )
        reply = await assistant.respond(history, content, context=session.context)
        assistant_message = await self.messages.insert(
            ChatMessage(
                session_id=session_id,
                user_id=user.id or "",
                role=ChatRole.ASSISTANT,
                content=reply.content,
                tool_calls=reply.tool_calls,
                data=reply.data,
                model=reply.model,
                tokens=reply.tokens,
                latency_ms=reply.latency_ms,
            )
        )
        fields: dict[str, Any] = {}
        if session.message_count == 0 and session.title == "New research session":
            fields["title"] = content.strip()[:80]
        await self.sessions.touch(session_id, messages_added=2, model=reply.model)
        if fields:
            await self.sessions.update(session_id, fields)
        refreshed = await self.sessions.get(session_id) or session
        return ChatReply(
            session=SessionOut.from_model(refreshed),
            user_message=MessageOut.from_model(user_message),
            assistant_message=MessageOut.from_model(assistant_message),
        )

    async def quick_ask(self, user: User, content: str) -> ChatReply:
        session = await self.create_session(user, title=None, context={})
        return await self.send_message(user, session.id or "", content)

    # ------------------------------------------------------------ insights
    async def list_insights(self, period: InsightPeriod | None, *, limit: int) -> list[InsightOut]:
        rows = await self.insights.latest(str(period) if period else None, limit=limit)
        return [InsightOut.from_model(r) for r in rows]

    async def refresh_insights(self) -> list[InsightOut]:
        now = utcnow()
        results = []
        for period in (InsightPeriod.MONTHLY, InsightPeriod.QUARTERLY):
            results.append(await compute_insight(self.db, self.settings, period, now, llm=self.llm))
            # Finalise the previous period if it has not been computed yet.
            previous_ref = results[-1].period_start - timedelta(days=1)
            existing = await self.insights.find_one(
                {"period": str(period), "period_start": {"$lte": previous_ref}, "in_progress": False}
            )
            if existing is None:
                results.append(await compute_insight(self.db, self.settings, period, previous_ref, llm=self.llm))
        return [InsightOut.from_model(r) for r in results]

    # ------------------------------------------------------------- library
    async def library(self, user: User) -> LibraryOut:
        sessions = await self.list_sessions(user)
        reports = (
            await self.db[Collections.REPORTS]
            .find(
                {"user_id": user.id},
                {"title": 1, "type": 1, "symbol": 1, "status": 1, "format": 1, "created_at": 1, "size_bytes": 1},
            )
            .sort("created_at", -1)
            .limit(20)
            .to_list(length=20)
        )
        trending = await run_tool(self.db, self.settings, "get_trending_topics", {"hours": 24, "limit": 8})
        latest = await self.insights.latest(None, limit=1)
        return LibraryOut(
            sessions=[SessionOut.from_model(s) for s in sessions[:20]],
            saved_reports=[
                SavedReportOut(
                    id=str(r["_id"]),
                    title=r.get("title") or "Report",
                    type=r.get("type", "report"),
                    symbol=r.get("symbol"),
                    status=r.get("status", "ready"),
                    format=r.get("format", "pdf"),
                    created_at=r["created_at"],
                    size_bytes=r.get("size_bytes"),
                )
                for r in reports
            ],
            trending_topics=trending.get("topics", []),
            latest_insight=InsightOut.from_model(latest[0]) if latest else None,
            assistant_mode=(await self._assistant()).mode,
        )

    async def pulse(self) -> ResearchPulse:
        now = utcnow()
        hour_ago = now - timedelta(hours=1)
        recent = await self.messages.find({"created_at": {"$gte": hour_ago}, "role": ChatRole.ASSISTANT}, limit=1000)
        latencies = [m.latency_ms for m in recent if m.latency_ms is not None]
        active_sessions = await self.sessions.count({"last_message_at": {"$gte": now - timedelta(hours=24)}})
        return ResearchPulse(
            assistant_mode=(await self._assistant()).mode,
            messages_last_hour=len(recent),
            active_sessions_24h=active_sessions,
            tool_calls_last_hour=sum(len(m.tool_calls) for m in recent),
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
            llm_tokens_total=self.llm.total_tokens,
            generated_at=now,
        )


__all__ = ["ResearchService", "shared_llm"]
