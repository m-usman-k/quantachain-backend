"""In-app notifications and push-device registration."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field

from app.api.deps import DB, CurrentUser, SettingsDep
from app.core.pagination import Page, Pagination
from app.modules.notifications.models import Notification, NotificationType, Severity
from app.modules.notifications.service import NotificationService

router = APIRouter(prefix="/notifications", tags=["notifications"])


def get_notification_service(db: DB, settings: SettingsDep) -> NotificationService:
    return NotificationService(db, settings)


Service = Annotated[NotificationService, Depends(get_notification_service)]


class NotificationPublic(BaseModel):
    id: str
    type: NotificationType
    title: str
    body: str
    severity: Severity
    data: dict[str, Any]
    read_at: datetime | None
    created_at: datetime

    @classmethod
    def from_model(cls, n: Notification) -> NotificationPublic:
        return cls(
            id=n.id or "",
            type=n.type,
            title=n.title,
            body=n.body,
            severity=n.severity,
            data=n.data,
            read_at=n.read_at,
            created_at=n.created_at,
        )


class DeviceRegisterRequest(BaseModel):
    token: str = Field(min_length=10, max_length=4096)
    platform: str = Field(default="android", pattern="^(android|ios|web)$")
    app_version: str | None = None


class DevicePublic(BaseModel):
    id: str
    platform: str
    app_version: str | None
    token_preview: str
    created_at: datetime
    last_seen_at: datetime | None


class UnreadCount(BaseModel):
    unread: int


@router.get("", response_model=Page[NotificationPublic], summary="My notifications")
async def list_notifications(
    user: CurrentUser,
    service: Service,
    page: Pagination,
    unread_only: Annotated[bool, Query()] = False,
) -> Page[NotificationPublic]:
    items, total = await service.list_for_user(user, page, unread_only=unread_only)
    return Page.build([NotificationPublic.from_model(n) for n in items], total, page)


@router.get("/unread-count", response_model=UnreadCount)
async def unread_count(user: CurrentUser, service: Service) -> UnreadCount:
    return UnreadCount(unread=await service.unread_count(user))


@router.post("/read-all", response_model=UnreadCount, summary="Mark every notification as read")
async def read_all(user: CurrentUser, service: Service) -> UnreadCount:
    await service.mark_all_read(user)
    return UnreadCount(unread=0)


@router.post("/{notification_id}/read", response_model=NotificationPublic)
async def mark_read(notification_id: str, user: CurrentUser, service: Service) -> NotificationPublic:
    return NotificationPublic.from_model(await service.mark_read(user, notification_id))


@router.delete("/{notification_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification(notification_id: str, user: CurrentUser, service: Service) -> None:
    await service.delete(user, notification_id)


@router.get("/devices", response_model=list[DevicePublic], summary="Registered push devices")
async def list_devices(user: CurrentUser, service: Service) -> list[DevicePublic]:
    return [
        DevicePublic(
            id=d.id or "",
            platform=d.platform,
            app_version=d.app_version,
            token_preview=f"{d.token[:6]}…{d.token[-4:]}",
            created_at=d.created_at,
            last_seen_at=d.last_seen_at,
        )
        for d in await service.list_devices(user)
    ]


@router.post(
    "/devices", response_model=DevicePublic, status_code=status.HTTP_201_CREATED, summary="Register an FCM device token"
)
async def register_device(payload: DeviceRegisterRequest, user: CurrentUser, service: Service) -> DevicePublic:
    d = await service.register_device(user, payload.token, payload.platform, payload.app_version)
    return DevicePublic(
        id=d.id or "",
        platform=d.platform,
        app_version=d.app_version,
        token_preview=f"{d.token[:6]}…{d.token[-4:]}",
        created_at=d.created_at,
        last_seen_at=d.last_seen_at,
    )


@router.delete("/devices/{token}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_device(token: str, user: CurrentUser, service: Service) -> None:
    await service.remove_device(user, token)
