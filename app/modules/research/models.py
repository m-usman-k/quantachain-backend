"""Module 8 persistence models: research chat and macro insights."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.db.models import Document


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatSession(Document):
    user_id: str
    title: str = "New research session"
    model: str = "heuristic"  # gpt-4o | heuristic
    message_count: int = 0
    last_message_at: datetime | None = None
    pinned: bool = False
    archived: bool = False
    context: dict[str, Any] = Field(default_factory=dict)  # e.g. {"symbols": ["BTC"]}


class ChatMessage(Document):
    session_id: str
    user_id: str
    role: ChatRole
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # tools the assistant invoked
    data: dict[str, Any] = Field(default_factory=dict)  # structured results for the UI
    model: str | None = None
    tokens: int | None = None
    latency_ms: float | None = None


class InsightPeriod(StrEnum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class MarketInsight(Document):
    period: InsightPeriod
    period_start: datetime
    period_end: datetime
    title: str
    summary: str
    highlights: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    performers: dict[str, Any] = Field(default_factory=dict)  # best / worst / table
    symbols: list[str] = Field(default_factory=list)
    generated_by: str = "template"  # template | gpt-4o
    in_progress: bool = False


__all__ = ["ChatMessage", "ChatRole", "ChatSession", "InsightPeriod", "MarketInsight"]
