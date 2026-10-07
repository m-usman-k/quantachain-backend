"""Cryptographic primitives: password hashing, JWTs, TOTP, symmetric encryption.

Kept free of FastAPI so it can be unit tested and reused by workers/scripts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt
import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings, get_settings

TokenType = Literal["access", "refresh", "mfa", "oauth_state", "report"]

_password_hasher = PasswordHasher()


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        # Run a hash anyway so timing does not reveal whether the account has a password.
        _password_hasher.hash(password)
        return False
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _password_hasher.check_needs_rehash(password_hash)


# ----------------------------------------------------------------------- JWT
class TokenError(Exception):
    """Raised for any invalid, expired or mistyped token."""


def create_token(
    subject: str,
    token_type: TokenType,
    *,
    expires_in: timedelta,
    settings: Settings | None = None,
    claims: dict[str, Any] | None = None,
    jti: str | None = None,
) -> str:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_in).timestamp()),
        "jti": jti or uuid.uuid4().hex,
    }
    if claims:
        payload.update(claims)
    return jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)


def decode_token(
    token: str, expected_type: TokenType | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp", "type", "jti"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid token") from exc
    if expected_type is not None and payload.get("type") != expected_type:
        raise TokenError(f"Expected a {expected_type} token")
    return payload  # type: ignore[no-any-return]


# ---------------------------------------------------------------------- TOTP
def generate_totp_secret() -> str:
    return pyotp.random_base32()


def totp_provisioning_uri(secret: str, account_name: str, issuer: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=account_name, issuer_name=issuer)


def verify_totp(secret: str, code: str, *, valid_window: int = 1) -> bool:
    code = code.strip().replace(" ", "")
    if not code.isdigit():
        return False
    return pyotp.TOTP(secret).verify(code, valid_window=valid_window)


def generate_recovery_codes(count: int = 8) -> list[str]:
    codes = []
    for _ in range(count):
        raw = secrets.token_hex(5)  # 10 hex chars
        codes.append(f"{raw[:5]}-{raw[5:]}")
    return codes


# ----------------------------------------------------------------- hashing
def sha256_hex(value: str | bytes) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


# -------------------------------------------------------------- encryption
class SecretBox:
    """Fernet (AES-128-CBC + HMAC) wrapper for secrets stored at rest."""

    def __init__(self, key: bytes | None = None) -> None:
        self._fernet = Fernet(key or get_settings().fernet_key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise ValueError("Could not decrypt secret (wrong ENCRYPTION_KEY?)") from exc


# ----------------------------------------------------------------- API keys
API_KEY_PREFIX = "qc_"


def generate_api_key() -> tuple[str, str]:
    """Return ``(plaintext_key, lookup_prefix)``. Only the hash is persisted."""
    random_part = base64.urlsafe_b64encode(secrets.token_bytes(30)).decode().rstrip("=")
    key = f"{API_KEY_PREFIX}{random_part}"
    return key, key[: len(API_KEY_PREFIX) + 8]


def hash_api_key(key: str) -> str:
    return sha256_hex(key)


__all__ = [
    "API_KEY_PREFIX",
    "SecretBox",
    "TokenError",
    "constant_time_equals",
    "create_token",
    "decode_token",
    "generate_api_key",
    "generate_recovery_codes",
    "generate_totp_secret",
    "hash_api_key",
    "hash_password",
    "password_needs_rehash",
    "sha256_hex",
    "totp_provisioning_uri",
    "verify_password",
    "verify_totp",
]
