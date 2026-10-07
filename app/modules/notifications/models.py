from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from app.db.models import Document


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class NotificationType(StrEnum):
    WHALE_ALERT = "whale_alert"
    FRAUD_ALERT = "fraud_alert"
    TRADE_SIGNAL = "trade_signal"
    ORDER_UPDATE = "order_update"
    PRICE_ALERT = "price_alert"
    SYSTEM = "system"
    REPORT_READY = "report_ready"


class DeviceToken(Document):
    user_id: str
    token: str
    platform: str = "android"  # android | ios | web
    app_version: str | None = None
    last_seen_at: datetime | None = None


class Notification(Document):
    user_id: str
    type: NotificationType
    title: str
    body: str
    severity: Severity = Severity.INFO
    data: dict[str, Any] = Field(default_factory=dict)
    read_at: datetime | None = None
    push_status: str | None = None  # sent | failed | skipped | disabled


__all__ = ["DeviceToken", "Notification", "NotificationType", "Severity"]
