"""Module 7 - administration, monitoring and control plane."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError, ValidationFailedError
from app.core.logging import log_buffer
from app.core.metrics import metrics
from app.core.pagination import PageParams
from app.core.timeutils import interval_seconds, parse_range, utcnow
from app.db.collections import Collections
from app.db.mongo import mongo
from app.modules.admin.controls import ModelControl, ModelControlService
from app.modules.admin.monitoring import SystemMonitor, collect_system_metrics
from app.modules.admin.schemas import (
    AdminOverview,
    ApiHealth,
    DatabaseHealth,
    HealthDashboard,
    JobOut,
    MetricPoint,
    ModelControlOut,
    ProviderOut,
    UserAdminOut,
)
from app.modules.auth.models import AuditLog, Role, User
from app.modules.auth.repository import ApiKeyRepository, AuditLogRepository, RefreshTokenRepository, UserRepository

STALE_HEARTBEAT = timedelta(seconds=75)
DATA_VOLUME_COLLECTIONS = (
    Collections.USERS,
    Collections.CANDLES,
    Collections.PRICE_TICKS,
    Collections.NEWS_ARTICLES,
    Collections.SOCIAL_POSTS,
    Collections.WHALE_TRANSFERS,
    Collections.PREDICTIONS,
    Collections.FRAUD_ALERTS,
    Collections.ORDERS,
    Collections.AUDIT_LOGS,
)


class AdminService:
    def __init__(
        self, db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None, *, actor: User | None = None
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.actor = actor
        self.users = UserRepository(db)
        self.audit = AuditLogRepository(db)
        self.refresh_tokens = RefreshTokenRepository(db)
        self.api_keys = ApiKeyRepository(db)
        self.controls = ModelControlService(db)

    async def _record(self, action: str, *, resource: str, resource_id: str | None = None, **metadata: Any) -> None:
        await self.audit.append(
            action=action,
            resource=resource,
            resource_id=resource_id,
            actor_id=self.actor.id if self.actor else None,
            actor_email=self.actor.email if self.actor else None,
            metadata=metadata,
        )

    # ------------------------------------------------------------- system
    async def system_metrics(self, monitor: SystemMonitor | None) -> dict[str, Any]:
        lag = await monitor.measure_loop_lag() if monitor else None
        return collect_system_metrics(loop_lag_ms=lag)

    async def metrics_history(self, *, range_: str, interval: str, role: str = "api") -> list[MetricPoint]:
        since = utcnow() - parse_range(range_)
        bucket = interval_seconds(interval)
        pipeline = [
            {"$match": {"ts": {"$gte": since}, "meta.role": role}},
            {
                "$group": {
                    "_id": {"$dateTrunc": {"date": "$ts", "unit": "second", "binSize": bucket}},
                    "cpu_percent": {"$avg": "$cpu_percent"},
                    "memory_percent": {"$avg": "$memory_percent"},
                    "process_rss_bytes": {"$avg": "$process_rss_bytes"},
                    "event_loop_lag_ms": {"$max": "$event_loop_lag_ms"},
                    "latency_p95_ms": {"$max": "$latency_p95_ms"},
                    "requests_total": {"$max": "$requests_total"},
                    "error_5xx": {"$max": "$error_5xx"},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        cursor = await self.db[Collections.SYSTEM_METRICS].aggregate(pipeline)
        return [MetricPoint(ts=row["_id"], **{k: v for k, v in row.items() if k != "_id"}) async for row in cursor]

    async def uptime_pct_24h(self) -> float | None:
        """Share of expected one-minute API samples present in the last 24h."""
        since = utcnow() - timedelta(hours=24)
        samples = await self.db[Collections.SYSTEM_METRICS].count_documents(
            {"ts": {"$gte": since}, "meta.role": "api", "healthy": True}
        )
        window_minutes = min(24 * 60, max(1.0, metrics.uptime_seconds / 60))
        if samples == 0:
            return None
        return round(min(100.0, 100 * samples / window_minutes), 3)

    async def health_dashboard(self) -> HealthDashboard:
        now = utcnow()
        ping = await mongo.ping_ms() if mongo.connected else None
        stats = None
        if ping is not None:
            try:
                stats = await mongo.stats()
            except Exception:  # pragma: no cover - restricted deployments
                stats = None
        sources = (
            await self.db[Collections.INGESTION_SOURCES]
            .find({}, {"_id": 0, "name": 1, "status": 1, "heartbeat_at": 1, "events_per_minute": 1, "enabled": 1})
            .to_list(length=100)
        )
        live = [s for s in sources if s.get("heartbeat_at") and now - s["heartbeat_at"] <= STALE_HEARTBEAT]
        jobs = await self.list_jobs()
        controls = await self.controls.list()
        volume: dict[str, int] = {}
        for name in DATA_VOLUME_COLLECTIONS:
            try:
                volume[name] = await self.db[name].estimated_document_count()
            except Exception:  # pragma: no cover
                volume[name] = 0
        snapshot = metrics.snapshot()
        failing_jobs = [j.name for j in jobs if j.consecutive_errors >= 3]
        status = "ok"
        if ping is None:
            status = "critical"
        elif failing_jobs or (snapshot["error_rate"] > 0.05 and metrics.requests_total > 50):
            status = "degraded"
        return HealthDashboard(
            status=status,
            api=ApiHealth(
                uptime_seconds=snapshot["uptime_seconds"],
                uptime_pct_24h=await self.uptime_pct_24h(),
                requests_total=snapshot["requests_total"],
                error_rate=snapshot["error_rate"],
                latency_ms=snapshot["latency_ms"],
                websocket_connections=snapshot["websocket_connections"],
                responses_by_class=snapshot["responses_by_class"],
            ),
            database=DatabaseHealth(
                connected=ping is not None, ping_ms=ping, version=await mongo.server_version(), stats=stats
            ),
            worker_online=bool(live),
            ingestion={
                "sources_total": len(sources),
                "sources_live": len(live),
                "connected": sum(1 for s in live if s.get("status") in ("connected", "active")),
                "events_per_minute": round(sum(float(s.get("events_per_minute") or 0) for s in live), 1),
                "sources": sources,
            },
            jobs={
                "total": len(jobs),
                "running": sum(1 for j in jobs if j.running),
                "paused": sum(1 for j in jobs if j.paused),
                "failing": failing_jobs,
            },
            models={
                "total": len(controls),
                "paused": [c.name for c in controls if c.paused],
                "overridden": [c.name for c in controls if c.override_active],
            },
            providers=[ProviderOut(**p) for p in snapshot["providers"]],
            data_volume=volume,
            generated_at=now,
        )

    async def overview(self) -> AdminOverview:
        now = utcnow()
        sources = (
            await self.db[Collections.INGESTION_SOURCES]
            .find({}, {"status": 1, "heartbeat_at": 1, "events_per_minute": 1})
            .to_list(length=100)
        )
        live = [s for s in sources if s.get("heartbeat_at") and now - s["heartbeat_at"] <= STALE_HEARTBEAT]
        total_users = await self.users.count({})
        month_ago = now - timedelta(days=30)
        users_before = await self.users.count({"created_at": {"$lt": month_ago}})
        growth = round(100 * (total_users - users_before) / users_before, 1) if users_before else None
        active_24h = await self.users.count({"last_login_at": {"$gte": now - timedelta(hours=24)}})
        alerts_24h = await self.db[Collections.FRAUD_ALERTS].count_documents(
            {"created_at": {"$gte": now - timedelta(hours=24)}, "acknowledged_at": None}
        )
        return AdminOverview(
            sources_connected=sum(1 for s in live if s.get("status") in ("connected", "active")),
            sources_total=len(sources),
            events_per_minute=round(sum(float(s.get("events_per_minute") or 0) for s in live), 1),
            uptime_pct_24h=await self.uptime_pct_24h(),
            registered_users=total_users,
            users_growth_30d_pct=growth,
            active_users_24h=active_24h,
            open_alerts_24h=alerts_24h,
            generated_at=now,
        )

    # --------------------------------------------------------------- jobs
    async def list_jobs(self) -> list[JobOut]:
        rows = await self.db[Collections.SCHEDULER_JOBS].find({}).sort("name", 1).to_list(length=200)
        return [JobOut(**{k: v for k, v in row.items() if k in JobOut.model_fields}) for row in rows]

    async def set_job_paused(self, name: str, paused: bool) -> JobOut:
        result = await self.db[Collections.SCHEDULER_JOBS].find_one_and_update(
            {"name": name}, {"$set": {"paused": paused, "updated_at": utcnow()}}, return_document=True
        )
        if result is None:
            raise NotFoundError(f"Job '{name}' not found")
        await self._record("job.paused" if paused else "job.resumed", resource="job", resource_id=name)
        return JobOut(**{k: v for k, v in result.items() if k in JobOut.model_fields})

    async def request_job_run(self, name: str) -> JobOut:
        result = await self.db[Collections.SCHEDULER_JOBS].find_one_and_update(
            {"name": name}, {"$set": {"run_requested_at": utcnow(), "updated_at": utcnow()}}, return_document=True
        )
        if result is None:
            raise NotFoundError(f"Job '{name}' not found")
        await self._record("job.run_requested", resource="job", resource_id=name)
        return JobOut(**{k: v for k, v in result.items() if k in JobOut.model_fields})

    # ------------------------------------------------------------- models
    @staticmethod
    def _control_out(control: ModelControl) -> ModelControlOut:
        return ModelControlOut(
            name=control.name,
            module=control.module,
            description=control.description,
            paused=control.paused,
            paused_by=control.paused_by,
            paused_at=control.paused_at,
            pause_reason=control.pause_reason,
            override=control.override,
            override_active=control.override_active,
            override_by=control.override_by,
            override_expires_at=control.override_expires_at,
            last_run_at=control.last_run_at,
            last_run_status=control.last_run_status,
            runs=control.runs,
        )

    async def list_models(self) -> list[ModelControlOut]:
        return [self._control_out(c) for c in await self.controls.list()]

    async def pause_model(self, name: str, *, paused: bool, reason: str | None) -> ModelControlOut:
        control = await self.controls.set_paused(
            name, paused, actor=self.actor.email if self.actor else None, reason=reason
        )
        await self._record(
            "model.paused" if paused else "model.resumed", resource="model", resource_id=name, reason=reason
        )
        return self._control_out(control)

    async def override_model(
        self, name: str, override: dict[str, Any] | None, *, expires_in_minutes: int | None
    ) -> ModelControlOut:
        expires_at = utcnow() + timedelta(minutes=expires_in_minutes) if (override and expires_in_minutes) else None
        control = await self.controls.set_override(
            name, override, actor=self.actor.email if self.actor else None, expires_at=expires_at
        )
        await self._record(
            "model.override_set" if override else "model.override_cleared",
            resource="model",
            resource_id=name,
            override=override,
        )
        return self._control_out(control)

    # -------------------------------------------------------------- users
    async def _user_out(self, user: User) -> UserAdminOut:
        return UserAdminOut(
            id=user.id or "",
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            mfa_enabled=user.mfa.enabled,
            oauth_providers=user.oauth_providers,
            api_keys=await self.api_keys.count({"user_id": user.id, "revoked_at": None}),
            login_count=user.login_count,
            last_login_at=user.last_login_at,
            created_at=user.created_at,
            locked_until=user.locked_until,
        )

    async def list_users(
        self, params: PageParams, *, query: str | None, role: Role | None, active: bool | None
    ) -> tuple[list[UserAdminOut], int]:
        filters: dict[str, Any] = {}
        if query:
            filters["$or"] = [
                {"email": {"$regex": query, "$options": "i"}},
                {"full_name": {"$regex": query, "$options": "i"}},
            ]
        if role is not None:
            filters["role"] = role
        if active is not None:
            filters["is_active"] = active
        users, total = await self.users.paginate(filters, params, sort=[("created_at", DESCENDING)])
        return [await self._user_out(u) for u in users], total

    async def get_user(self, user_id: str) -> UserAdminOut:
        return await self._user_out(await self.users.get_or_raise(user_id))

    async def update_user(
        self, user_id: str, *, role: Role | None, is_active: bool | None, full_name: str | None
    ) -> UserAdminOut:
        user = await self.users.get_or_raise(user_id)
        if self.actor and user.id == self.actor.id and (role == Role.USER or is_active is False):
            raise ValidationFailedError("You cannot demote or deactivate your own account")
        changes: dict[str, Any] = {}
        if role is not None and role != user.role:
            changes["role"] = role
        if is_active is not None and is_active != user.is_active:
            changes["is_active"] = is_active
        if full_name is not None:
            changes["full_name"] = full_name
        if not changes:
            return await self._user_out(user)
        inc = {"token_version": 1} if ("role" in changes or changes.get("is_active") is False) else None
        updated = await self.users.update(user_id, changes, inc=inc)
        if changes.get("is_active") is False:
            await self.refresh_tokens.revoke_all_for_user(user_id)
        await self._record(
            "admin.user_updated", resource="user", resource_id=user_id, changes={k: str(v) for k, v in changes.items()}
        )
        return await self._user_out(updated or user)

    async def reset_user_mfa(self, user_id: str) -> UserAdminOut:
        user = await self.users.get_or_raise(user_id)
        updated = await self.users.update(
            user_id,
            {
                "mfa.enabled": False,
                "mfa.secret_encrypted": None,
                "mfa.pending_secret_encrypted": None,
                "mfa.recovery_code_hashes": [],
                "mfa.enabled_at": None,
            },
        )
        await self._record("admin.user_mfa_reset", resource="user", resource_id=user_id)
        return await self._user_out(updated or user)

    async def revoke_user_sessions(self, user_id: str) -> int:
        await self.users.get_or_raise(user_id)
        revoked = await self.refresh_tokens.revoke_all_for_user(user_id)
        await self.users.update(user_id, inc={"token_version": 1})
        await self._record("admin.user_sessions_revoked", resource="user", resource_id=user_id, revoked=revoked)
        return revoked

    # --------------------------------------------------------------- audit
    @staticmethod
    def audit_filters(
        *,
        actor_id: str | None,
        action: str | None,
        resource: str | None,
        since: datetime | None,
        until: datetime | None,
        outcome: str | None,
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if actor_id:
            filters["actor_id"] = actor_id
        if action:
            filters["action"] = {"$regex": f"^{action}", "$options": "i"}
        if resource:
            filters["resource"] = resource
        if outcome:
            filters["outcome"] = outcome
        if since or until:
            filters["created_at"] = {k: v for k, v in (("$gte", since), ("$lte", until)) if v}
        return filters

    async def list_audit(self, params: PageParams, filters: dict[str, Any]) -> tuple[list[AuditLog], int]:
        return await self.audit.paginate(filters, params, sort=[("seq", DESCENDING)])

    async def export_audit(self, filters: dict[str, Any], fmt: str) -> AsyncIterator[bytes]:
        cursor = self.audit.col.find(filters).sort("seq", 1)
        if fmt == "json":
            yield b"["
            first = True
            async for row in cursor:
                row["_id"] = str(row["_id"])
                chunk = json.dumps(row, default=str).encode()
                yield chunk if first else b"," + chunk
                first = False
            yield b"]"
            return
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "seq",
                "created_at",
                "actor_id",
                "actor_email",
                "action",
                "resource",
                "resource_id",
                "outcome",
                "ip",
                "user_agent",
                "metadata",
                "prev_hash",
                "hash",
            ]
        )
        async for row in cursor:
            writer.writerow(
                [
                    row.get("seq"),
                    row.get("created_at"),
                    row.get("actor_id"),
                    row.get("actor_email"),
                    row.get("action"),
                    row.get("resource"),
                    row.get("resource_id"),
                    row.get("outcome"),
                    row.get("ip"),
                    row.get("user_agent"),
                    json.dumps(row.get("metadata", {}), default=str),
                    row.get("prev_hash"),
                    row.get("hash"),
                ]
            )
            yield buffer.getvalue().encode()
            buffer.seek(0)
            buffer.truncate(0)
        if buffer.tell():
            yield buffer.getvalue().encode()

    async def verify_audit(self) -> dict[str, Any]:
        result = await self.audit.verify_chain()
        await self._record("audit.verified", resource="audit", valid=result["valid"], checked=result["checked"])
        return result

    # ---------------------------------------------------------------- logs
    @staticmethod
    def recent_logs(limit: int, level: str | None) -> list[dict[str, Any]]:
        return log_buffer.recent(limit, min_level=level)

    async def export_logs(self, fmt: str, limit: int) -> bytes:
        records = log_buffer.recent(limit)
        ingestion = (
            await self.db[Collections.INGESTION_LOGS].find({}).sort("_id", -1).limit(limit).to_list(length=limit)
        )
        for row in ingestion:
            records.append(
                {
                    "timestamp": row.get("ts").isoformat() if row.get("ts") else "",
                    "level": row.get("level", "info"),
                    "logger": f"ingestion.{row.get('source', '')}",
                    "event": row.get("message", ""),
                }
            )
        records.sort(key=lambda r: str(r.get("timestamp", "")))
        if fmt == "json":
            return json.dumps(records, default=str).encode()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["timestamp", "level", "logger", "event", "request_id"])
        for r in records:
            writer.writerow([r.get("timestamp"), r.get("level"), r.get("logger"), r.get("event"), r.get("request_id")])
        return buffer.getvalue().encode()


__all__ = ["DATA_VOLUME_COLLECTIONS", "STALE_HEARTBEAT", "AdminService"]
