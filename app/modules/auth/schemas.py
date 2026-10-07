"""Request/response schemas for the auth API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.modules.auth.models import Role, User, UserPreferences


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=120)

    @field_validator("password")
    @classmethod
    def _password_strength(cls, value: str) -> str:
        if value.lower() == value or not any(ch.isdigit() for ch in value):
            raise ValueError("Password must contain an uppercase letter and a digit")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    id: str
    email: str
    full_name: str | None
    avatar_url: str | None
    role: Role
    is_active: bool
    is_verified: bool
    mfa_enabled: bool
    oauth_providers: list[str]
    preferences: UserPreferences
    created_at: datetime
    last_login_at: datetime | None

    @classmethod
    def from_user(cls, user: User) -> UserPublic:
        return cls(
            id=user.id or "",
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            mfa_enabled=user.mfa.enabled,
            oauth_providers=user.oauth_providers,
            preferences=user.preferences,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")


class AuthResponse(BaseModel):
    """Login result: either tokens, or an MFA challenge to complete first."""

    mfa_required: bool = False
    mfa_token: str | None = Field(default=None, description="Exchange with a TOTP code at /auth/mfa/verify")
    tokens: TokenPair | None = None
    user: UserPublic | None = None


class MFAVerifyRequest(BaseModel):
    mfa_token: str
    code: str = Field(min_length=6, max_length=16, description="TOTP code or a recovery code")


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None
    all_devices: bool = False


class UserUpdateRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=120)
    avatar_url: str | None = Field(default=None, max_length=500)
    preferences: UserPreferences | None = None


class PasswordChangeRequest(BaseModel):
    current_password: str | None = Field(default=None, description="Omit for OAuth-only accounts without a password")
    new_password: str = Field(min_length=8, max_length=128)


class MFASetupResponse(BaseModel):
    secret: str
    otpauth_uri: str
    issuer: str


class MFACodeRequest(BaseModel):
    code: str = Field(min_length=6, max_length=16)


class MFADisableRequest(BaseModel):
    code: str | None = None
    password: str | None = None


class RecoveryCodesResponse(BaseModel):
    recovery_codes: list[str]
    message: str = "Store these codes safely; they are shown only once."


class OAuthStartResponse(BaseModel):
    provider: str
    authorization_url: str
    state: str


class SessionPublic(BaseModel):
    id: str
    created_at: datetime
    expires_at: datetime
    user_agent: str | None
    ip: str | None
    current: bool = False


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    expires_in_days: int | None = Field(default=None, ge=1, le=3650)


class ApiKeyPublic(BaseModel):
    id: str
    name: str
    prefix: str
    scopes: list[str]
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreatedResponse(ApiKeyPublic):
    key: str = Field(description="Full API key; shown only once")


class AuditLogPublic(BaseModel):
    id: str
    seq: int
    actor_id: str | None
    actor_email: str | None
    action: str
    resource: str
    resource_id: str | None
    outcome: str
    metadata: dict[str, Any]
    ip: str | None
    created_at: datetime
    hash: str


class AuditChainVerification(BaseModel):
    valid: bool
    checked: int
    broken_at_seq: int | None
    reason: str | None


__all__ = [name for name in globals() if name[0].isupper()]
