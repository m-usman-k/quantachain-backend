"""Module 1 - Authentication & Access Control endpoints."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import RedirectResponse

from app.api.deps import DB, CurrentUser, SettingsDep, client_ip
from app.core.pagination import Page, Pagination
from app.core.rate_limit import rate_limited
from app.core.security import decode_token
from app.modules.auth.oauth import PROVIDERS
from app.modules.auth.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyPublic,
    AuditLogPublic,
    AuthResponse,
    LoginRequest,
    LogoutRequest,
    MFACodeRequest,
    MFADisableRequest,
    MFASetupResponse,
    MFAVerifyRequest,
    OAuthStartResponse,
    PasswordChangeRequest,
    RecoveryCodesResponse,
    RefreshRequest,
    RegisterRequest,
    SessionPublic,
    TokenPair,
    UserPublic,
    UserUpdateRequest,
)
from app.modules.auth.service import AuthService, RequestMeta

router = APIRouter(prefix="/auth", tags=["auth"])


def get_auth_service(request: Request, db: DB, settings: SettingsDep) -> AuthService:
    meta = RequestMeta(ip=client_ip(request), user_agent=request.headers.get("user-agent"))
    return AuthService(db, settings, meta)


Service = Annotated[AuthService, Depends(get_auth_service)]

login_limiter = rate_limited(20)


# ------------------------------------------------------------- registration
@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(login_limiter)],
    summary="Create an account with email and password",
)
async def register(payload: RegisterRequest, service: Service) -> AuthResponse:
    user, tokens = await service.register(payload.email, payload.password, payload.full_name)
    return AuthResponse(tokens=tokens, user=UserPublic.from_user(user))


@router.post(
    "/login",
    response_model=AuthResponse,
    dependencies=[Depends(login_limiter)],
    summary="Log in; returns tokens or an MFA challenge",
)
async def login(payload: LoginRequest, service: Service) -> AuthResponse:
    return await service.login(payload.email, payload.password)


@router.post(
    "/mfa/verify",
    response_model=AuthResponse,
    dependencies=[Depends(login_limiter)],
    summary="Complete an MFA challenge with a TOTP or recovery code",
)
async def mfa_verify(payload: MFAVerifyRequest, service: Service) -> AuthResponse:
    return await service.verify_mfa(payload.mfa_token, payload.code)


@router.post("/refresh", response_model=TokenPair, summary="Rotate a refresh token")
async def refresh(payload: RefreshRequest, service: Service) -> TokenPair:
    return await service.refresh(payload.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke the current or all sessions")
async def logout(payload: LogoutRequest, user: CurrentUser, service: Service) -> None:
    await service.logout(user, payload.refresh_token, all_devices=payload.all_devices)


# ------------------------------------------------------------------ profile
@router.get("/me", response_model=UserPublic, summary="Current user profile")
async def me(user: CurrentUser) -> UserPublic:
    return UserPublic.from_user(user)


@router.patch("/me", response_model=UserPublic, summary="Update profile and preferences")
async def update_me(payload: UserUpdateRequest, user: CurrentUser, service: Service) -> UserPublic:
    updated = await service.update_profile(
        user, full_name=payload.full_name, avatar_url=payload.avatar_url, preferences=payload.preferences
    )
    return UserPublic.from_user(updated)


@router.put("/password", status_code=status.HTTP_204_NO_CONTENT, summary="Change password (revokes other sessions)")
async def change_password(payload: PasswordChangeRequest, user: CurrentUser, service: Service) -> None:
    await service.change_password(user, payload.current_password, payload.new_password)


@router.get("/sessions", response_model=list[SessionPublic], summary="Active sessions (refresh tokens)")
async def sessions(request: Request, user: CurrentUser, service: Service) -> list[SessionPublic]:
    current_jti = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        try:
            current_jti = decode_token(auth_header[7:], "access").get("jti")
        except Exception:  # pragma: no cover - header already validated by dependency
            current_jti = None
    result = []
    for session in await service.sessions(user):
        result.append(
            SessionPublic(
                id=session.id or "",
                created_at=session.created_at,
                expires_at=session.expires_at,
                user_agent=session.user_agent,
                ip=session.ip,
                current=session.jti == current_jti,
            )
        )
    return result


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke one session")
async def revoke_session(session_id: str, user: CurrentUser, service: Service) -> None:
    await service.revoke_session(user, session_id)


# ---------------------------------------------------------------------- MFA
@router.post("/mfa/setup", response_model=MFASetupResponse, summary="Start TOTP enrolment (Google Authenticator)")
async def mfa_setup(user: CurrentUser, service: Service) -> MFASetupResponse:
    return await service.mfa_setup(user)


@router.post("/mfa/enable", response_model=RecoveryCodesResponse, summary="Confirm enrolment with a code")
async def mfa_enable(payload: MFACodeRequest, user: CurrentUser, service: Service) -> RecoveryCodesResponse:
    return RecoveryCodesResponse(recovery_codes=await service.mfa_enable(user, payload.code))


@router.post("/mfa/disable", status_code=status.HTTP_204_NO_CONTENT, summary="Disable MFA")
async def mfa_disable(payload: MFADisableRequest, user: CurrentUser, service: Service) -> None:
    await service.mfa_disable(user, code=payload.code, password=payload.password)


@router.post("/mfa/recovery-codes", response_model=RecoveryCodesResponse, summary="Regenerate recovery codes")
async def mfa_recovery_codes(payload: MFACodeRequest, user: CurrentUser, service: Service) -> RecoveryCodesResponse:
    return RecoveryCodesResponse(recovery_codes=await service.mfa_regenerate_recovery_codes(user, payload.code))


# -------------------------------------------------------------------- OAuth
@router.get("/oauth/providers", response_model=list[str], summary="OAuth providers configured on this server")
async def oauth_providers(settings: SettingsDep) -> list[str]:
    available = []
    if settings.google_client_id and settings.google_client_secret:
        available.append("google")
    if settings.github_client_id and settings.github_client_secret:
        available.append("github")
    return available


@router.get("/oauth/{provider}/start", response_model=OAuthStartResponse, summary="Begin OAuth 2.0 login")
async def oauth_start(
    provider: str,
    service: Service,
    mode: Annotated[
        str, Query(pattern="^(redirect|json)$", description="How the callback returns tokens")
    ] = "redirect",
) -> OAuthStartResponse:
    if provider not in PROVIDERS:
        from app.core.exceptions import ValidationFailedError

        raise ValidationFailedError(f"Unknown OAuth provider '{provider}'")
    url, state = service.oauth_start(provider, redirect_mode=mode)
    return OAuthStartResponse(provider=provider, authorization_url=url, state=state)


@router.get("/oauth/{provider}/callback", summary="OAuth 2.0 callback (code exchange)")
async def oauth_callback(provider: str, code: str, state: str, service: Service, settings: SettingsDep):  # type: ignore[no-untyped-def]
    tokens, user, mode = await service.oauth_callback(provider, code, state)
    if mode == "json":
        return AuthResponse(tokens=tokens, user=UserPublic.from_user(user))
    fragment = urlencode(
        {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "expires_in": tokens.expires_in,
            "provider": provider,
        }
    )
    return RedirectResponse(url=f"{settings.frontend_url.rstrip('/')}/auth/callback#{fragment}", status_code=302)


# ----------------------------------------------------------------- API keys
@router.get("/api-keys", response_model=list[ApiKeyPublic], summary="List my API keys")
async def list_api_keys(user: CurrentUser, service: Service) -> list[ApiKeyPublic]:
    return [ApiKeyPublic(**k.model_dump()) for k in await service.api_keys.list_for_user(user.id or "")]


@router.post(
    "/api-keys", response_model=ApiKeyCreatedResponse, status_code=status.HTTP_201_CREATED, summary="Create an API key"
)
async def create_api_key(payload: ApiKeyCreateRequest, user: CurrentUser, service: Service) -> ApiKeyCreatedResponse:
    return await service.create_api_key(user, payload)


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke an API key")
async def revoke_api_key(key_id: str, user: CurrentUser, service: Service) -> None:
    await service.revoke_api_key(user, key_id)


# ---------------------------------------------------------------- my audit
@router.get("/audit", response_model=Page[AuditLogPublic], summary="My audit trail")
async def my_audit(user: CurrentUser, service: Service, page: Pagination) -> Page[AuditLogPublic]:
    items, total = await service.audit.paginate({"actor_id": user.id}, page, sort=[("seq", -1)])
    return Page.build([AuditLogPublic(**i.model_dump()) for i in items], total, page)
