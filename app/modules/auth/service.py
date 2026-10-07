"""Authentication business logic."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
    ValidationFailedError,
)
from app.core.security import (
    SecretBox,
    TokenError,
    create_token,
    decode_token,
    generate_api_key,
    generate_recovery_codes,
    generate_totp_secret,
    hash_api_key,
    hash_password,
    sha256_hex,
    totp_provisioning_uri,
    verify_password,
    verify_totp,
)
from app.core.timeutils import utcnow
from app.modules.auth import oauth
from app.modules.auth.models import (
    ApiKey,
    AuditOutcome,
    OAuthAccount,
    RefreshToken,
    Role,
    User,
    UserPreferences,
)
from app.modules.auth.repository import (
    ApiKeyRepository,
    AuditLogRepository,
    RefreshTokenRepository,
    UserRepository,
)
from app.modules.auth.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    AuthResponse,
    MFASetupResponse,
    TokenPair,
    UserPublic,
)

logger = structlog.get_logger(__name__)

MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15


@dataclass
class RequestMeta:
    ip: str | None = None
    user_agent: str | None = None


class AuthService:
    def __init__(
        self,
        db: AsyncDatabase[dict[str, Any]],
        settings: Settings | None = None,
        meta: RequestMeta | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.meta = meta or RequestMeta()
        self.users = UserRepository(db)
        self.refresh_tokens = RefreshTokenRepository(db)
        self.audit = AuditLogRepository(db)
        self.api_keys = ApiKeyRepository(db)
        self.secrets = SecretBox(self.settings.fernet_key)

    # ------------------------------------------------------------- audit
    async def record(
        self,
        action: str,
        *,
        resource: str,
        actor: User | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        **metadata: Any,
    ) -> None:
        await self.audit.append(
            action=action,
            resource=resource,
            resource_id=resource_id,
            actor_id=actor.id if actor else None,
            actor_email=actor.email if actor else None,
            outcome=outcome,
            metadata=metadata,
            ip=self.meta.ip,
            user_agent=self.meta.user_agent,
        )

    # ------------------------------------------------------ registration
    async def register(self, email: str, password: str, full_name: str | None) -> tuple[User, TokenPair]:
        email = email.lower().strip()
        if await self.users.get_by_email(email):
            raise ConflictError("An account with this email already exists")
        role = await self._initial_role(email)
        user = User(email=email, full_name=full_name, password_hash=hash_password(password), role=role)
        try:
            user = await self.users.insert(user)
        except DuplicateKeyError as exc:
            raise ConflictError("An account with this email already exists") from exc
        await self.record("user.registered", resource="user", actor=user, resource_id=user.id, role=str(role))
        tokens = await self.issue_tokens(user)
        return user, tokens

    async def _initial_role(self, email: str) -> Role:
        if self.settings.first_admin_email and email == self.settings.first_admin_email.lower():
            return Role.ADMIN
        if self.settings.environment == "development" and await self.users.count() == 0:
            logger.warning("first_user_promoted_to_admin", email=email)
            return Role.ADMIN
        return Role.USER

    # ------------------------------------------------------------- login
    async def login(self, email: str, password: str) -> AuthResponse:
        user = await self.users.get_by_email(email)
        if user is None or not verify_password(password, user.password_hash):
            if user is not None:
                await self.users.record_failed_login(
                    user.id or "", max_attempts=MAX_FAILED_LOGINS, lock_minutes=LOCKOUT_MINUTES
                )
            await self.record(
                "user.login_failed", resource="user", actor=user, outcome=AuditOutcome.FAILURE, email=email.lower()
            )
            raise UnauthorizedError("Invalid email or password")
        if user.is_locked:
            raise ForbiddenError("Account temporarily locked after too many failed attempts")
        if not user.is_active:
            raise ForbiddenError("This account has been deactivated")
        if user.mfa.enabled:
            mfa_token = create_token(
                user.id or "",
                "mfa",
                expires_in=timedelta(minutes=self.settings.mfa_token_ttl_minutes),
                settings=self.settings,
            )
            return AuthResponse(mfa_required=True, mfa_token=mfa_token)
        return await self._complete_login(user, method="password")

    async def verify_mfa(self, mfa_token: str, code: str) -> AuthResponse:
        try:
            payload = decode_token(mfa_token, "mfa", settings=self.settings)
        except TokenError as exc:
            raise UnauthorizedError(str(exc)) from exc
        user = await self.users.get(payload["sub"])
        if user is None or not user.mfa.enabled or not user.mfa.secret_encrypted:
            raise UnauthorizedError("MFA challenge is no longer valid")
        if not await self._check_mfa_code(user, code):
            await self.record("mfa.failed", resource="user", actor=user, outcome=AuditOutcome.FAILURE)
            raise UnauthorizedError("Invalid verification code")
        return await self._complete_login(user, method="mfa")

    async def _complete_login(self, user: User, *, method: str) -> AuthResponse:
        await self.users.record_login(user.id or "")
        tokens = await self.issue_tokens(user)
        await self.record("user.login", resource="user", actor=user, resource_id=user.id, method=method)
        return AuthResponse(tokens=tokens, user=UserPublic.from_user(user))

    # ------------------------------------------------------------ tokens
    async def issue_tokens(self, user: User, *, family_id: str | None = None) -> TokenPair:
        access_ttl = timedelta(minutes=self.settings.access_token_ttl_minutes)
        refresh_ttl = timedelta(days=self.settings.refresh_token_ttl_days)
        access = create_token(
            user.id or "",
            "access",
            expires_in=access_ttl,
            settings=self.settings,
            claims={"role": str(user.role), "ver": user.token_version, "email": user.email},
        )
        jti = uuid.uuid4().hex
        refresh = create_token(user.id or "", "refresh", expires_in=refresh_ttl, settings=self.settings, jti=jti)
        await self.refresh_tokens.insert(
            RefreshToken(
                user_id=user.id or "",
                jti=jti,
                family_id=family_id or jti,
                token_hash=sha256_hex(refresh),
                expires_at=utcnow() + refresh_ttl,
                user_agent=self.meta.user_agent,
                ip=self.meta.ip,
            )
        )
        return TokenPair(access_token=access, refresh_token=refresh, expires_in=int(access_ttl.total_seconds()))

    async def refresh(self, refresh_token: str) -> TokenPair:
        try:
            payload = decode_token(refresh_token, "refresh", settings=self.settings)
        except TokenError as exc:
            raise UnauthorizedError(str(exc)) from exc
        stored = await self.refresh_tokens.get_by_jti(payload["jti"])
        if stored is None or stored.token_hash != sha256_hex(refresh_token):
            raise UnauthorizedError("Unknown refresh token")
        if stored.revoked_at is not None:
            # Reuse of a rotated token: assume theft and kill the whole family.
            revoked = await self.refresh_tokens.revoke_family(stored.family_id)
            user = await self.users.get(stored.user_id)
            await self.record(
                "token.reuse_detected",
                resource="session",
                actor=user,
                resource_id=stored.family_id,
                outcome=AuditOutcome.FAILURE,
                revoked_sessions=revoked,
            )
            raise UnauthorizedError("Refresh token has been revoked")
        if stored.expires_at <= utcnow():
            raise UnauthorizedError("Refresh token has expired")
        user = await self.users.get(stored.user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError("Account unavailable")
        tokens = await self.issue_tokens(user, family_id=stored.family_id)
        new_jti = decode_token(tokens.refresh_token, "refresh", settings=self.settings)["jti"]
        await self.refresh_tokens.revoke(stored.jti, replaced_by=new_jti)
        return tokens

    async def logout(self, user: User, refresh_token: str | None, *, all_devices: bool) -> int:
        revoked = 0
        if all_devices:
            revoked = await self.refresh_tokens.revoke_all_for_user(user.id or "")
            await self.users.update(user.id or "", inc={"token_version": 1})
        elif refresh_token:
            try:
                payload = decode_token(refresh_token, "refresh", settings=self.settings)
                stored = await self.refresh_tokens.get_by_jti(payload["jti"])
                if stored and stored.user_id == user.id and stored.revoked_at is None:
                    await self.refresh_tokens.revoke(stored.jti)
                    revoked = 1
            except TokenError:
                pass
        await self.record("user.logout", resource="session", actor=user, all_devices=all_devices, revoked=revoked)
        return revoked

    async def sessions(self, user: User) -> list[RefreshToken]:
        return await self.refresh_tokens.active_sessions(user.id or "")

    async def revoke_session(self, user: User, session_id: str) -> None:
        session = await self.refresh_tokens.get(session_id)
        if session is None or session.user_id != user.id:
            raise NotFoundError("Session not found")
        await self.refresh_tokens.revoke(session.jti)
        await self.record("session.revoked", resource="session", actor=user, resource_id=session_id)

    # ----------------------------------------------------------- profile
    async def update_profile(
        self, user: User, *, full_name: str | None, avatar_url: str | None, preferences: UserPreferences | None
    ) -> User:
        changes: dict[str, Any] = {}
        if full_name is not None:
            changes["full_name"] = full_name
        if avatar_url is not None:
            changes["avatar_url"] = avatar_url
        if preferences is not None:
            changes["preferences"] = preferences.model_dump()
        if not changes:
            return user
        updated = await self.users.update(user.id or "", changes)
        await self.record("user.profile_updated", resource="user", actor=user, fields=sorted(changes))
        return updated or user

    async def change_password(self, user: User, current_password: str | None, new_password: str) -> None:
        if user.password_hash and not verify_password(current_password or "", user.password_hash):
            await self.record("password.change_failed", resource="user", actor=user, outcome=AuditOutcome.FAILURE)
            raise UnauthorizedError("Current password is incorrect")
        if len(new_password) < self.settings.password_min_length:
            raise ValidationFailedError(f"Password must be at least {self.settings.password_min_length} characters")
        await self.users.update(user.id or "", {"password_hash": hash_password(new_password)}, inc={"token_version": 1})
        await self.refresh_tokens.revoke_all_for_user(user.id or "")
        await self.record("password.changed", resource="user", actor=user)

    # --------------------------------------------------------------- MFA
    async def mfa_setup(self, user: User) -> MFASetupResponse:
        secret = generate_totp_secret()
        await self.users.update(user.id or "", {"mfa.pending_secret_encrypted": self.secrets.encrypt(secret)})
        return MFASetupResponse(
            secret=secret,
            otpauth_uri=totp_provisioning_uri(secret, user.email, self.settings.mfa_issuer),
            issuer=self.settings.mfa_issuer,
        )

    async def mfa_enable(self, user: User, code: str) -> list[str]:
        if not user.mfa.pending_secret_encrypted:
            raise ValidationFailedError("Start MFA setup first")
        secret = self.secrets.decrypt(user.mfa.pending_secret_encrypted)
        if not verify_totp(secret, code):
            raise UnauthorizedError("Invalid verification code")
        codes = generate_recovery_codes()
        await self.users.update(
            user.id or "",
            {
                "mfa.enabled": True,
                "mfa.secret_encrypted": user.mfa.pending_secret_encrypted,
                "mfa.pending_secret_encrypted": None,
                "mfa.recovery_code_hashes": [sha256_hex(c) for c in codes],
                "mfa.enabled_at": utcnow(),
            },
        )
        await self.record("mfa.enabled", resource="user", actor=user)
        return codes

    async def mfa_disable(self, user: User, *, code: str | None, password: str | None) -> None:
        if not user.mfa.enabled:
            raise ValidationFailedError("MFA is not enabled")
        authorised = False
        if code:
            authorised = await self._check_mfa_code(user, code, consume_recovery=False)
        if not authorised and password and user.password_hash:
            authorised = verify_password(password, user.password_hash)
        if not authorised:
            raise UnauthorizedError("Provide a valid verification code or password")
        await self.users.update(
            user.id or "",
            {
                "mfa.enabled": False,
                "mfa.secret_encrypted": None,
                "mfa.pending_secret_encrypted": None,
                "mfa.recovery_code_hashes": [],
                "mfa.enabled_at": None,
            },
        )
        await self.record("mfa.disabled", resource="user", actor=user)

    async def mfa_regenerate_recovery_codes(self, user: User, code: str) -> list[str]:
        if not user.mfa.enabled or not await self._check_mfa_code(user, code, consume_recovery=False):
            raise UnauthorizedError("Invalid verification code")
        codes = generate_recovery_codes()
        await self.users.update(user.id or "", {"mfa.recovery_code_hashes": [sha256_hex(c) for c in codes]})
        await self.record("mfa.recovery_codes_regenerated", resource="user", actor=user)
        return codes

    async def _check_mfa_code(self, user: User, code: str, *, consume_recovery: bool = True) -> bool:
        if not user.mfa.secret_encrypted:
            return False
        secret = self.secrets.decrypt(user.mfa.secret_encrypted)
        if verify_totp(secret, code):
            return True
        code_hash = sha256_hex(code.strip().lower())
        if code_hash in user.mfa.recovery_code_hashes:
            if consume_recovery:
                await self.users.update(user.id or "", pull={"mfa.recovery_code_hashes": code_hash})
                await self.record("mfa.recovery_code_used", resource="user", actor=user)
            return True
        return False

    # ------------------------------------------------------------- OAuth
    def oauth_start(self, provider: str, *, redirect_mode: str = "redirect") -> tuple[str, str]:
        state = create_token(
            secrets.token_urlsafe(16),
            "oauth_state",
            expires_in=timedelta(minutes=10),
            settings=self.settings,
            claims={"provider": provider, "mode": redirect_mode},
        )
        return oauth.build_authorization_url(self.settings, provider, state), state

    async def oauth_callback(self, provider: str, code: str, state: str) -> tuple[TokenPair, User, str]:
        try:
            payload = decode_token(state, "oauth_state", settings=self.settings)
        except TokenError as exc:
            raise UnauthorizedError("Invalid or expired OAuth state") from exc
        if payload.get("provider") != provider:
            raise UnauthorizedError("OAuth state does not match provider")
        profile = await oauth.exchange_code(self.settings, provider, code)

        user = await self.users.get_by_oauth(provider, profile.provider_user_id)
        created = False
        if user is None and profile.email:
            user = await self.users.get_by_email(profile.email)
            if user is not None:
                if not profile.email_verified:
                    raise ForbiddenError("Verify your email with the provider before linking this account")
                await self.users.update(
                    user.id or "",
                    {"is_verified": True},
                    push={
                        "oauth_accounts": OAuthAccount(
                            provider=provider, provider_user_id=profile.provider_user_id, email=profile.email
                        ).model_dump()
                    },
                )
                await self.record("oauth.linked", resource="user", actor=user, provider=provider)
        if user is None:
            if not profile.email:
                raise ValidationFailedError("The provider did not share an email address")
            user = await self.users.insert(
                User(
                    email=profile.email,
                    full_name=profile.full_name,
                    avatar_url=profile.avatar_url,
                    is_verified=profile.email_verified,
                    role=await self._initial_role(profile.email),
                    oauth_accounts=[
                        OAuthAccount(provider=provider, provider_user_id=profile.provider_user_id, email=profile.email)
                    ],
                )
            )
            created = True
            await self.record("user.registered", resource="user", actor=user, resource_id=user.id, provider=provider)
        user = await self.users.get(user.id or "") or user
        if not user.is_active:
            raise ForbiddenError("This account has been deactivated")
        await self.users.record_login(user.id or "")
        tokens = await self.issue_tokens(user)
        await self.record("user.login", resource="user", actor=user, method=f"oauth:{provider}", new_user=created)
        return tokens, user, str(payload.get("mode", "redirect"))

    # ---------------------------------------------------------- API keys
    async def create_api_key(self, user: User, request: ApiKeyCreateRequest) -> ApiKeyCreatedResponse:
        plaintext, prefix = generate_api_key()
        expires_at = utcnow() + timedelta(days=request.expires_in_days) if request.expires_in_days else None
        key = await self.api_keys.insert(
            ApiKey(
                user_id=user.id or "",
                name=request.name,
                prefix=prefix,
                key_hash=hash_api_key(plaintext),
                scopes=request.scopes,
                expires_at=expires_at,
            )
        )
        await self.record("apikey.created", resource="api_key", actor=user, resource_id=key.id, name=key.name)
        return ApiKeyCreatedResponse(
            id=key.id or "",
            name=key.name,
            prefix=key.prefix,
            scopes=key.scopes,
            created_at=key.created_at,
            last_used_at=None,
            expires_at=key.expires_at,
            revoked_at=None,
            key=plaintext,
        )

    async def revoke_api_key(self, user: User, key_id: str) -> None:
        key = await self.api_keys.get(key_id)
        if key is None or key.user_id != user.id:
            raise NotFoundError("API key not found")
        await self.api_keys.update(key_id, {"revoked_at": utcnow()})
        await self.record("apikey.revoked", resource="api_key", actor=user, resource_id=key_id)


__all__ = ["AuthService", "RequestMeta"]
