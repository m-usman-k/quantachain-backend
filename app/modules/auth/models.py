"""Persistence models for users, sessions, API keys and the audit trail."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.core.timeutils import utcnow
from app.db.models import Document


class Role(StrEnum):
    ADMIN = "admin"
    USER = "user"


class OAuthAccount(BaseModel):
    provider: str
    provider_user_id: str
    email: str | None = None
    linked_at: datetime = Field(default_factory=utcnow)


class MFASettings(BaseModel):
    enabled: bool = False
    secret_encrypted: str | None = None
    pending_secret_encrypted: str | None = None
    recovery_code_hashes: list[str] = Field(default_factory=list)
    enabled_at: datetime | None = None


class NotificationPreferences(BaseModel):
    push_enabled: bool = True
    whale_alerts: bool = True
    fraud_alerts: bool = True
    trade_signals: bool = True
    order_updates: bool = True
    email_alerts: bool = False
    min_whale_usd: float = 10_000_000.0


class UserPreferences(BaseModel):
    timezone: str = "UTC"
    default_exchange: str = "binance"
    favorite_symbols: list[str] = Field(default_factory=lambda: ["BTC", "ETH"])
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences)


class User(Document):
    email: str
    full_name: str | None = None
    avatar_url: str | None = None
    password_hash: str | None = None
    role: Role = Role.USER
    is_active: bool = True
    is_verified: bool = False
    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)
    mfa: MFASettings = Field(default_factory=MFASettings)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    token_version: int = 0  # bump to invalidate every issued access token
    last_login_at: datetime | None = None
    login_count: int = 0
    failed_login_attempts: int = 0
    locked_until: datetime | None = None

    @property
    def is_locked(self) -> bool:
        return self.locked_until is not None and self.locked_until > utcnow()

    @property
    def oauth_providers(self) -> list[str]:
        return [account.provider for account in self.oauth_accounts]


class RefreshToken(Document):
    user_id: str
    jti: str
    family_id: str  # all tokens descending from one login share a family
    token_hash: str
    expires_at: datetime
    revoked_at: datetime | None = None
    replaced_by_jti: str | None = None
    user_agent: str | None = None
    ip: str | None = None


class AuditOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class AuditLog(Document):
    seq: int
    actor_id: str | None = None
    actor_email: str | None = None
    action: str
    resource: str
    resource_id: str | None = None
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    metadata: dict[str, Any] = Field(default_factory=dict)
    ip: str | None = None
    user_agent: str | None = None
    prev_hash: str
    hash: str


class ApiKey(Document):
    user_id: str
    name: str
    prefix: str
    key_hash: str
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= utcnow()


__all__ = [
    "ApiKey",
    "AuditLog",
    "AuditOutcome",
    "MFASettings",
    "NotificationPreferences",
    "OAuthAccount",
    "RefreshToken",
    "Role",
    "User",
    "UserPreferences",
]
