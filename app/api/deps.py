"""Shared FastAPI dependencies: database, authentication and authorization."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import TokenError, decode_token, hash_api_key
from app.db.mongo import get_db
from app.modules.auth.models import Role, User
from app.modules.auth.repository import ApiKeyRepository, UserRepository

DB = Annotated[AsyncDatabase[dict[str, Any]], Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

_bearer = HTTPBearer(auto_error=False)


async def _user_from_bearer(token: str, db: AsyncDatabase[dict[str, Any]]) -> User:
    try:
        payload = decode_token(token, "access")
    except TokenError as exc:
        raise UnauthorizedError(str(exc)) from exc
    user = await UserRepository(db).get(payload["sub"])
    if user is None:
        raise UnauthorizedError("User no longer exists")
    if user.token_version != payload.get("ver", 0):
        raise UnauthorizedError("Session has been revoked")
    return user


async def _user_from_api_key(api_key: str, db: AsyncDatabase[dict[str, Any]]) -> User:
    key = await ApiKeyRepository(db).find_one({"key_hash": hash_api_key(api_key), "revoked_at": None})
    if key is None or key.is_expired:
        raise UnauthorizedError("Invalid API key")
    user = await UserRepository(db).get(key.user_id)
    if user is None:
        raise UnauthorizedError("Invalid API key")
    await ApiKeyRepository(db).touch(key.id or "")
    return user


async def get_optional_user(
    request: Request,
    db: DB,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> User | None:
    user: User | None = None
    if credentials is not None and credentials.scheme.lower() == "bearer":
        user = await _user_from_bearer(credentials.credentials, db)
    elif x_api_key:
        user = await _user_from_api_key(x_api_key, db)
    if user is not None:
        request.state.user_id = user.id
    return user


async def get_current_user(user: Annotated[User | None, Depends(get_optional_user)]) -> User:
    if user is None:
        raise UnauthorizedError("Authentication required")
    if not user.is_active:
        raise ForbiddenError("This account has been deactivated")
    return user


def require_roles(*roles: Role):  # type: ignore[no-untyped-def]
    async def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        if user.role not in roles:
            raise ForbiddenError("Insufficient permissions")
        return user

    return dependency


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
AdminUser = Annotated[User, Depends(require_roles(Role.ADMIN))]


def client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


__all__ = [
    "DB",
    "AdminUser",
    "CurrentUser",
    "OptionalUser",
    "SettingsDep",
    "client_ip",
    "get_current_user",
    "get_optional_user",
    "require_roles",
]
