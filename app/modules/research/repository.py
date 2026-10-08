from __future__ import annotations

from pymongo import ASCENDING, DESCENDING

from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.modules.research.models import ChatMessage, ChatSession, MarketInsight


class ChatSessionRepository(Repository[ChatSession]):
    collection_name = Collections.CHAT_SESSIONS
    model = ChatSession
    not_found_message = "Chat session not found"

    async def for_user(self, user_id: str, *, include_archived: bool = False) -> list[ChatSession]:
        query = {"user_id": user_id}
        if not include_archived:
            query["archived"] = False  # type: ignore[assignment]
        return await self.find(query, sort=[("pinned", DESCENDING), ("last_message_at", DESCENDING)], limit=100)

    async def touch(self, session_id: str, *, messages_added: int, model: str | None = None) -> None:
        fields = {"last_message_at": utcnow()}
        if model:
            fields["model"] = model  # type: ignore[assignment]
        await self.update(session_id, fields, inc={"message_count": messages_added})


class ChatMessageRepository(Repository[ChatMessage]):
    collection_name = Collections.CHAT_MESSAGES
    model = ChatMessage

    async def history(self, session_id: str, *, limit: int = 40) -> list[ChatMessage]:
        rows = await self.find({"session_id": session_id}, sort=[("created_at", DESCENDING)], limit=limit)
        rows.reverse()
        return rows

    async def all_for_session(self, session_id: str) -> list[ChatMessage]:
        return await self.find({"session_id": session_id}, sort=[("created_at", ASCENDING)], limit=1000)


class MarketInsightRepository(Repository[MarketInsight]):
    collection_name = Collections.MARKET_INSIGHTS
    model = MarketInsight
    not_found_message = "Insight not found"

    async def latest(self, period: str | None = None, *, limit: int = 6) -> list[MarketInsight]:
        query = {"period": period} if period else {}
        return await self.find(query, sort=[("period_start", DESCENDING)], limit=limit)


__all__ = ["ChatMessageRepository", "ChatSessionRepository", "MarketInsightRepository"]
