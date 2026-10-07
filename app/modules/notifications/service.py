"""Delivers notifications in-app (Mongo + WebSocket) and as FCM push messages."""

from __future__ import annotations

from typing import Any

import structlog
from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings, get_settings
from app.core.events import Topics, event_bus
from app.core.exceptions import NotFoundError
from app.core.pagination import PageParams
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.integrations.fcm import FCMClient
from app.modules.auth.models import User
from app.modules.auth.repository import UserRepository
from app.modules.notifications.models import DeviceToken, Notification, NotificationType, Severity

logger = structlog.get_logger(__name__)


class DeviceTokenRepository(Repository[DeviceToken]):
    collection_name = Collections.DEVICE_TOKENS
    model = DeviceToken


class NotificationRepository(Repository[Notification]):
    collection_name = Collections.NOTIFICATIONS
    model = Notification
    not_found_message = "Notification not found"


# Maps notification types to the user preference that gates them.
PREFERENCE_FOR_TYPE: dict[NotificationType, str] = {
    NotificationType.WHALE_ALERT: "whale_alerts",
    NotificationType.FRAUD_ALERT: "fraud_alerts",
    NotificationType.TRADE_SIGNAL: "trade_signals",
    NotificationType.ORDER_UPDATE: "order_updates",
}


class NotificationService:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, fcm: FCMClient | None = None
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.devices = DeviceTokenRepository(db)
        self.notifications = NotificationRepository(db)
        self.users = UserRepository(db)
        self.fcm = fcm or FCMClient(self.settings)

    # ------------------------------------------------------------- devices
    async def register_device(self, user: User, token: str, platform: str, app_version: str | None) -> DeviceToken:
        existing = await self.devices.find_one({"token": token})
        if existing:
            updated = await self.devices.update(
                existing.id or "",
                {"user_id": user.id, "platform": platform, "app_version": app_version, "last_seen_at": utcnow()},
            )
            return updated or existing
        try:
            return await self.devices.insert(
                DeviceToken(
                    user_id=user.id or "",
                    token=token,
                    platform=platform,
                    app_version=app_version,
                    last_seen_at=utcnow(),
                )
            )
        except DuplicateKeyError:
            return await self.devices.find_one({"token": token})  # type: ignore[return-value]

    async def remove_device(self, user: User, token: str) -> bool:
        return (await self.devices.delete_many({"user_id": user.id, "token": token})) > 0

    async def list_devices(self, user: User) -> list[DeviceToken]:
        return await self.devices.find({"user_id": user.id}, sort=[("created_at", DESCENDING)])

    # ------------------------------------------------------------ delivery
    async def notify(
        self,
        user_id: str,
        *,
        type: NotificationType,
        title: str,
        body: str,
        severity: Severity = Severity.INFO,
        data: dict[str, Any] | None = None,
        push: bool = True,
    ) -> Notification:
        notification = await self.notifications.insert(
            Notification(user_id=user_id, type=type, title=title, body=body, severity=severity, data=data or {})
        )
        payload = {
            "id": notification.id,
            "user_id": user_id,
            "type": str(type),
            "title": title,
            "body": body,
            "severity": str(severity),
            "data": data or {},
            "created_at": notification.created_at.isoformat(),
        }
        event_bus.publish(Topics.ALERT, payload)

        status = "disabled"
        if push:
            status = await self._push(
                user_id, title, body, {"type": str(type), **{k: str(v) for k, v in (data or {}).items()}}
            )
        await self.notifications.update(notification.id or "", {"push_status": status})
        notification.push_status = status
        return notification

    async def notify_subscribers(
        self,
        *,
        type: NotificationType,
        title: str,
        body: str,
        severity: Severity = Severity.INFO,
        data: dict[str, Any] | None = None,
        extra_filter: dict[str, Any] | None = None,
        limit: int = 5000,
    ) -> int:
        """Notify every active user whose preferences opt in to this notification type."""
        pref = PREFERENCE_FOR_TYPE.get(type)
        query: dict[str, Any] = {"is_active": True}
        if pref:
            query[f"preferences.notifications.{pref}"] = True
        if extra_filter:
            query.update(extra_filter)
        count = 0
        cursor = self.users.col.find(query, {"_id": 1}).limit(limit)
        async for row in cursor:
            await self.notify(str(row["_id"]), type=type, title=title, body=body, severity=severity, data=data)
            count += 1
        return count

    async def _push(self, user_id: str, title: str, body: str, data: dict[str, str]) -> str:
        user = await self.users.get(user_id)
        if user is None or not user.preferences.notifications.push_enabled:
            return "disabled"
        devices = await self.devices.find({"user_id": user_id})
        if not devices:
            return "skipped"
        result = await self.fcm.send([d.token for d in devices], title=title, body=body, data=data)
        if result.invalid_tokens:
            await self.devices.delete_many({"token": {"$in": result.invalid_tokens}})
        if result.skipped:
            return "skipped"
        return "sent" if result.sent else "failed"

    # --------------------------------------------------------------- inbox
    async def list_for_user(
        self, user: User, params: PageParams, *, unread_only: bool = False
    ) -> tuple[list[Notification], int]:
        query: dict[str, Any] = {"user_id": user.id}
        if unread_only:
            query["read_at"] = None
        return await self.notifications.paginate(query, params, sort=[("created_at", DESCENDING)])

    async def unread_count(self, user: User) -> int:
        return await self.notifications.count({"user_id": user.id, "read_at": None})

    async def mark_read(self, user: User, notification_id: str) -> Notification:
        notification = await self.notifications.get(notification_id)
        if notification is None or notification.user_id != user.id:
            raise NotFoundError("Notification not found")
        if notification.read_at is None:
            notification = await self.notifications.update(notification_id, {"read_at": utcnow()}) or notification
        return notification

    async def mark_all_read(self, user: User) -> int:
        return await self.notifications.update_many({"user_id": user.id, "read_at": None}, {"read_at": utcnow()})

    async def delete(self, user: User, notification_id: str) -> None:
        deleted = await self.notifications.delete_many({"_id": self._oid(notification_id), "user_id": user.id})
        if not deleted:
            raise NotFoundError("Notification not found")

    @staticmethod
    def _oid(value: str) -> Any:
        from app.db.models import to_object_id

        return to_object_id(value)


__all__ = ["PREFERENCE_FOR_TYPE", "DeviceTokenRepository", "NotificationRepository", "NotificationService"]
