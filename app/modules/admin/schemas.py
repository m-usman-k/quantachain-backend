"""API schemas for Module 7."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.auth.models import Role


class SystemMetricsOut(BaseModel):
    host: str
    pid: int
    cpu_percent: float
    cpu_count: int
    load_avg: list[float] | None
    memory: dict[str, Any]
    disk: dict[str, Any]
    process: dict[str, Any]
    event_loop_lag_ms: float | None
    python: str
    platform: str
    sampled_at: datetime


class MetricPoint(BaseModel):
    ts: datetime
    cpu_percent: float | None
    memory_percent: float | None
    process_rss_bytes: float | None
    event_loop_lag_ms: float | None
    latency_p95_ms: float | None
    requests_total: float | None
    error_5xx: float | None


class JobOut(BaseModel):
    name: str
    module: str | None = None
    description: str | None = None
    interval_seconds: float | None = None
    enabled: bool = True
    paused: bool = False
    running: bool = False
    last_run_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    last_duration_ms: float | None = None
    next_run_at: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    last_result: Any = None
    host: str | None = None
    run_requested_at: datetime | None = None
    updated_at: datetime | None = None


class ModelControlOut(BaseModel):
    name: str
    module: str
    description: str
    paused: bool
    paused_by: str | None
    paused_at: datetime | None
    pause_reason: str | None
    override: dict[str, Any] | None
    override_active: bool
    override_by: str | None
    override_expires_at: datetime | None
    last_run_at: datetime | None
    last_run_status: str | None
    runs: int


class PauseRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=300)


class OverrideRequest(BaseModel):
    override: dict[str, Any] = Field(description="Manual values returned instead of the model output")
    expires_in_minutes: int | None = Field(default=None, ge=1, le=7 * 24 * 60)


class UserAdminOut(BaseModel):
    id: str
    email: str
    full_name: str | None
    role: Role
    is_active: bool
    is_verified: bool
    mfa_enabled: bool
    oauth_providers: list[str]
    api_keys: int
    login_count: int
    last_login_at: datetime | None
    created_at: datetime
    locked_until: datetime | None


class UserAdminUpdate(BaseModel):
    role: Role | None = None
    is_active: bool | None = None
    full_name: str | None = Field(default=None, max_length=120)


class AuditLogAdminOut(BaseModel):
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
    user_agent: str | None
    created_at: datetime
    prev_hash: str
    hash: str


class LogRecordOut(BaseModel):
    timestamp: str
    level: str
    logger: str
    event: str
    request_id: str | None = None
    source: str | None = None
    job: str | None = None


class ProviderOut(BaseModel):
    name: str
    calls: int
    errors: int
    error_rate: float
    avg_latency_ms: float | None
    last_latency_ms: float | None
    last_call_at: datetime | None
    last_error: str | None
    last_error_at: datetime | None
    rate_limit_remaining: int | None
    rate_limit_reset_at: datetime | None


class ApiHealth(BaseModel):
    uptime_seconds: float
    uptime_pct_24h: float | None
    requests_total: int
    error_rate: float
    latency_ms: dict[str, float | None]
    websocket_connections: int
    responses_by_class: dict[str, int]


class DatabaseHealth(BaseModel):
    connected: bool
    ping_ms: float | None
    version: str | None
    stats: dict[str, Any] | None


class HealthDashboard(BaseModel):
    status: str
    api: ApiHealth
    database: DatabaseHealth
    worker_online: bool
    ingestion: dict[str, Any]
    jobs: dict[str, Any]
    models: dict[str, Any]
    providers: list[ProviderOut]
    data_volume: dict[str, int]
    generated_at: datetime


class AdminOverview(BaseModel):
    sources_connected: int
    sources_total: int
    events_per_minute: float
    uptime_pct_24h: float | None
    registered_users: int
    users_growth_30d_pct: float | None
    active_users_24h: int
    open_alerts_24h: int
    generated_at: datetime


class AlertConfigOut(BaseModel):
    smtp_configured: bool
    recipients: list[str]
    error_alert_threshold: int
    error_alert_window_seconds: int
    error_alert_cooldown_seconds: int
    fcm_configured: bool


class AlertTestOut(BaseModel):
    email_sent: bool
    admins_notified: int


__all__ = [name for name in globals() if name[0].isupper()]
