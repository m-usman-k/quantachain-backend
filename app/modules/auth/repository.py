"""Repositories for the auth module."""

from __future__ import annotations

import json
from typing import Any

from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

from app.core.security import sha256_hex
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.repository import Repository
from app.modules.auth.models import ApiKey, AuditLog, AuditOutcome, RefreshToken, User

GENESIS_HASH = "0" * 64


class UserRepository(Repository[User]):
    collection_name = Collections.USERS
    model = User
    not_found_message = "User not found"

    async def get_by_email(self, email: str) -> User | None:
        return await self.find_one({"email": email.lower().strip()})

    async def get_by_oauth(self, provider: str, provider_user_id: str) -> User | None:
        return await self.find_one(
            {"oauth_accounts": {"$elemMatch": {"provider": provider, "provider_user_id": provider_user_id}}}
        )

    async def record_login(self, user_id: str) -> None:
        await self.col.update_one(
            {"_id": self._oid(user_id)},
            {
                "$set": {"last_login_at": utcnow(), "failed_login_attempts": 0, "locked_until": None},
                "$inc": {"login_count": 1},
            },
        )

    async def record_failed_login(self, user_id: str, *, max_attempts: int, lock_minutes: int) -> None:
        from datetime import timedelta

        user = await self.get(user_id)
        if user is None:
            return
        attempts = user.failed_login_attempts + 1
        update: dict[str, Any] = {"failed_login_attempts": attempts}
        if attempts >= max_attempts:
            update["locked_until"] = utcnow() + timedelta(minutes=lock_minutes)
            update["failed_login_attempts"] = 0
        await self.col.update_one({"_id": self._oid(user_id)}, {"$set": update})

    @staticmethod
    def _oid(value: str) -> Any:
        from app.db.models import to_object_id

        return to_object_id(value)


class RefreshTokenRepository(Repository[RefreshToken]):
    collection_name = Collections.REFRESH_TOKENS
    model = RefreshToken

    async def get_by_jti(self, jti: str) -> RefreshToken | None:
        return await self.find_one({"jti": jti})

    async def revoke(self, jti: str, *, replaced_by: str | None = None) -> None:
        await self.col.update_one(
            {"jti": jti, "revoked_at": None},
            {"$set": {"revoked_at": utcnow(), "replaced_by_jti": replaced_by}},
        )

    async def revoke_family(self, family_id: str) -> int:
        result = await self.col.update_many(
            {"family_id": family_id, "revoked_at": None}, {"$set": {"revoked_at": utcnow()}}
        )
        return result.modified_count

    async def revoke_all_for_user(self, user_id: str) -> int:
        result = await self.col.update_many(
            {"user_id": user_id, "revoked_at": None}, {"$set": {"revoked_at": utcnow()}}
        )
        return result.modified_count

    async def active_sessions(self, user_id: str) -> list[RefreshToken]:
        return await self.find(
            {"user_id": user_id, "revoked_at": None, "expires_at": {"$gt": utcnow()}},
            sort=[("created_at", DESCENDING)],
        )


class AuditLogRepository(Repository[AuditLog]):
    """Append-only, hash-chained audit trail.

    Each entry stores the hash of the previous entry and its own hash over its
    canonical content, so any modification or deletion of a past record is
    detectable by :meth:`verify_chain`.
    """

    collection_name = Collections.AUDIT_LOGS
    model = AuditLog

    @staticmethod
    def compute_hash(prev_hash: str, entry: dict[str, Any]) -> str:
        canonical = json.dumps(entry, sort_keys=True, default=str, separators=(",", ":"))
        return sha256_hex(prev_hash + canonical)

    def _canonical_fields(self, log: AuditLog) -> dict[str, Any]:
        return {
            "seq": log.seq,
            "actor_id": log.actor_id,
            "actor_email": log.actor_email,
            "action": log.action,
            "resource": log.resource,
            "resource_id": log.resource_id,
            "outcome": str(log.outcome),
            "metadata": log.metadata,
            "ip": log.ip,
            "user_agent": log.user_agent,
            "created_at": log.created_at.isoformat(),
        }

    async def append(
        self,
        *,
        action: str,
        resource: str,
        actor_id: str | None = None,
        actor_email: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        metadata: dict[str, Any] | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> AuditLog:
        for _attempt in range(5):
            last = await self.find_one({}, sort=[("seq", DESCENDING)])
            seq = (last.seq + 1) if last else 1
            prev_hash = last.hash if last else GENESIS_HASH
            # BSON stores datetimes at millisecond precision; hash exactly what will be read back.
            now = utcnow()
            log = AuditLog(
                created_at=now.replace(microsecond=(now.microsecond // 1000) * 1000),
                seq=seq,
                actor_id=actor_id,
                actor_email=actor_email,
                action=action,
                resource=resource,
                resource_id=resource_id,
                outcome=outcome,
                metadata=metadata or {},
                ip=ip,
                user_agent=user_agent,
                prev_hash=prev_hash,
                hash="",
            )
            log.hash = self.compute_hash(prev_hash, self._canonical_fields(log))
            try:
                return await self.insert(log)
            except DuplicateKeyError:
                continue  # another writer took this seq; recompute on the new tail
        raise RuntimeError("Could not append to the audit log after several attempts")

    async def verify_chain(self, *, limit: int | None = None) -> dict[str, Any]:
        cursor = self.col.find({}).sort("seq", 1)
        if limit:
            cursor = cursor.limit(limit)
        prev_hash = GENESIS_HASH
        expected_seq = 1
        checked = 0
        async for raw in cursor:
            log = self._parse(raw)
            if log.seq != expected_seq or log.prev_hash != prev_hash:
                return {"valid": False, "checked": checked, "broken_at_seq": log.seq, "reason": "chain_link"}
            if self.compute_hash(prev_hash, self._canonical_fields(log)) != log.hash:
                return {"valid": False, "checked": checked, "broken_at_seq": log.seq, "reason": "content_hash"}
            prev_hash = log.hash
            expected_seq += 1
            checked += 1
        return {"valid": True, "checked": checked, "broken_at_seq": None, "reason": None}


class ApiKeyRepository(Repository[ApiKey]):
    collection_name = Collections.API_KEYS
    model = ApiKey
    not_found_message = "API key not found"

    async def touch(self, key_id: str) -> None:
        if not key_id:
            return
        from app.db.models import to_object_id

        await self.col.update_one({"_id": to_object_id(key_id)}, {"$set": {"last_used_at": utcnow()}})

    async def list_for_user(self, user_id: str) -> list[ApiKey]:
        return await self.find({"user_id": user_id}, sort=[("created_at", DESCENDING)])


__all__ = [
    "GENESIS_HASH",
    "ApiKeyRepository",
    "AuditLogRepository",
    "RefreshTokenRepository",
    "UserRepository",
]
