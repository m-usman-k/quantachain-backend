"""Module 7 - Admin Dashboard & System Monitoring endpoints (admin role required)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import Response, StreamingResponse

from app.api.deps import DB, AdminUser, SettingsDep
from app.core.pagination import Page, Pagination
from app.integrations.fcm import FCMClient
from app.modules.admin.monitoring import SystemMonitor
from app.modules.admin.schemas import (
    AdminOverview,
    AlertConfigOut,
    AlertTestOut,
    AuditLogAdminOut,
    HealthDashboard,
    JobOut,
    LogRecordOut,
    MetricPoint,
    ModelControlOut,
    OverrideRequest,
    PauseRequest,
    ProviderOut,
    SystemMetricsOut,
    UserAdminOut,
    UserAdminUpdate,
)
from app.modules.admin.service import AdminService
from app.modules.auth.models import Role
from app.modules.auth.schemas import AuditChainVerification

router = APIRouter(prefix="/admin", tags=["admin"])


def get_admin_service(db: DB, settings: SettingsDep, admin: AdminUser) -> AdminService:
    return AdminService(db, settings, actor=admin)


Service = Annotated[AdminService, Depends(get_admin_service)]


def _monitor(request: Request) -> SystemMonitor | None:
    return getattr(request.app.state, "system_monitor", None)


# -------------------------------------------------------------- dashboards
@router.get("/overview", response_model=AdminOverview, summary="Admin landing cards (nodes, throughput, uptime, users)")
async def overview(service: Service) -> AdminOverview:
    return await service.overview()


@router.get("/health", response_model=HealthDashboard, summary="Centralised health dashboard")
async def health(service: Service) -> HealthDashboard:
    return await service.health_dashboard()


@router.get("/system/metrics", response_model=SystemMetricsOut, summary="Live CPU / RAM / process metrics")
async def system_metrics(request: Request, service: Service) -> SystemMetricsOut:
    return SystemMetricsOut(**await service.system_metrics(_monitor(request)))


@router.get("/system/metrics/history", response_model=list[MetricPoint], summary="Sampled system metrics series")
async def metrics_history(
    service: Service,
    range: Annotated[str, Query(pattern=r"^\d+[mhdw]$")] = "24h",
    interval: Annotated[str, Query(pattern=r"^\d+[mhd]$")] = "5m",
    role: Annotated[str, Query(pattern="^(api|worker)$")] = "api",
) -> list[MetricPoint]:
    return await service.metrics_history(range_=range, interval=interval, role=role)


@router.get("/providers", response_model=list[ProviderOut], summary="Third-party API usage and rate limits")
async def providers(_: Service) -> list[ProviderOut]:
    from app.core.metrics import metrics

    return [ProviderOut(**p) for p in metrics.snapshot()["providers"]]


# -------------------------------------------------------------------- jobs
@router.get("/jobs", response_model=list[JobOut], summary="Scheduler jobs and their state")
async def jobs(service: Service) -> list[JobOut]:
    return await service.list_jobs()


@router.post("/jobs/{name}/pause", response_model=JobOut)
async def pause_job(name: str, service: Service) -> JobOut:
    return await service.set_job_paused(name, True)


@router.post("/jobs/{name}/resume", response_model=JobOut)
async def resume_job(name: str, service: Service) -> JobOut:
    return await service.set_job_paused(name, False)


@router.post("/jobs/{name}/run", response_model=JobOut, summary="Ask the worker to run a job immediately")
async def run_job(name: str, service: Service) -> JobOut:
    return await service.request_job_run(name)


# ------------------------------------------------------------------ models
@router.get("/models", response_model=list[ModelControlOut], summary="AI model controls (pause / override)")
async def models(service: Service) -> list[ModelControlOut]:
    return await service.list_models()


@router.post("/models/{name}/pause", response_model=ModelControlOut)
async def pause_model(name: str, payload: PauseRequest, service: Service) -> ModelControlOut:
    return await service.pause_model(name, paused=True, reason=payload.reason)


@router.post("/models/{name}/resume", response_model=ModelControlOut)
async def resume_model(name: str, service: Service) -> ModelControlOut:
    return await service.pause_model(name, paused=False, reason=None)


@router.put(
    "/models/{name}/override", response_model=ModelControlOut, summary="Pin a manual value instead of model output"
)
async def override_model(name: str, payload: OverrideRequest, service: Service) -> ModelControlOut:
    return await service.override_model(name, payload.override, expires_in_minutes=payload.expires_in_minutes)


@router.delete("/models/{name}/override", response_model=ModelControlOut)
async def clear_override(name: str, service: Service) -> ModelControlOut:
    return await service.override_model(name, None, expires_in_minutes=None)


# ------------------------------------------------------------------- users
@router.get("/users", response_model=Page[UserAdminOut], summary="User & access management")
async def users(
    service: Service,
    page: Pagination,
    q: Annotated[str | None, Query()] = None,
    role: Annotated[Role | None, Query()] = None,
    active: Annotated[bool | None, Query()] = None,
) -> Page[UserAdminOut]:
    items, total = await service.list_users(page, query=q, role=role, active=active)
    return Page.build(items, total, page)


@router.get("/users/{user_id}", response_model=UserAdminOut)
async def user(user_id: str, service: Service) -> UserAdminOut:
    return await service.get_user(user_id)


@router.patch("/users/{user_id}", response_model=UserAdminOut, summary="Change role, activation or name")
async def update_user(user_id: str, payload: UserAdminUpdate, service: Service) -> UserAdminOut:
    return await service.update_user(
        user_id, role=payload.role, is_active=payload.is_active, full_name=payload.full_name
    )


@router.post("/users/{user_id}/reset-mfa", response_model=UserAdminOut)
async def reset_mfa(user_id: str, service: Service) -> UserAdminOut:
    return await service.reset_user_mfa(user_id)


@router.post("/users/{user_id}/revoke-sessions", response_model=dict[str, int])
async def revoke_sessions(user_id: str, service: Service) -> dict[str, int]:
    return {"revoked": await service.revoke_user_sessions(user_id)}


# ------------------------------------------------------------------- audit
@router.get("/audit-logs", response_model=Page[AuditLogAdminOut], summary="Immutable audit trail")
async def audit_logs(
    service: Service,
    page: Pagination,
    actor_id: Annotated[str | None, Query()] = None,
    action: Annotated[str | None, Query(description="Prefix match, e.g. 'user.' or 'mfa.'")] = None,
    resource: Annotated[str | None, Query()] = None,
    outcome: Annotated[str | None, Query(pattern="^(success|failure)$")] = None,
    since: Annotated[datetime | None, Query()] = None,
    until: Annotated[datetime | None, Query()] = None,
) -> Page[AuditLogAdminOut]:
    filters = AdminService.audit_filters(
        actor_id=actor_id, action=action, resource=resource, since=since, until=until, outcome=outcome
    )
    items, total = await service.list_audit(page, filters)
    return Page.build([AuditLogAdminOut(**i.model_dump()) for i in items], total, page)


@router.get("/audit-logs/verify", response_model=AuditChainVerification, summary="Verify the audit hash chain")
async def verify_audit(service: Service) -> AuditChainVerification:
    return AuditChainVerification(**await service.verify_audit())


@router.get("/audit-logs/export", summary="Export audit logs as CSV or JSON")
async def export_audit(
    service: Service,
    format: Annotated[str, Query(pattern="^(csv|json)$")] = "csv",
    since: Annotated[datetime | None, Query()] = None,
    until: Annotated[datetime | None, Query()] = None,
    action: Annotated[str | None, Query()] = None,
) -> StreamingResponse:
    filters = AdminService.audit_filters(
        actor_id=None, action=action, resource=None, since=since, until=until, outcome=None
    )
    media = "text/csv" if format == "csv" else "application/json"
    filename = f"audit-logs-{datetime.now().strftime('%Y%m%d-%H%M%S')}.{format}"
    return StreamingResponse(
        service.export_audit(filters, format),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# -------------------------------------------------------------------- logs
@router.get("/logs/recent", response_model=list[LogRecordOut], summary="Recent application log records")
async def recent_logs(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    level: Annotated[str | None, Query(pattern="^(debug|info|warning|error|critical)$")] = None,
) -> list[LogRecordOut]:
    return [
        LogRecordOut(**{k: v for k, v in r.items() if k in LogRecordOut.model_fields})
        for r in service.recent_logs(limit, level)
    ]


@router.get("/logs/export", summary="Export recent application + ingestion logs")
async def export_logs(
    service: Service,
    format: Annotated[str, Query(pattern="^(csv|json)$")] = "csv",
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> Response:
    content = await service.export_logs(format, limit)
    media = "text/csv" if format == "csv" else "application/json"
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="logs.{format}"'})


# ------------------------------------------------------------------ alerts
@router.get("/alerts/config", response_model=AlertConfigOut, summary="Automated alert configuration")
async def alert_config(settings: SettingsDep, _: Service) -> AlertConfigOut:
    return AlertConfigOut(
        smtp_configured=bool(settings.smtp_host),
        recipients=[_mask_email(e) for e in settings.admin_alert_emails],
        error_alert_threshold=settings.error_alert_threshold,
        error_alert_window_seconds=settings.error_alert_window_seconds,
        error_alert_cooldown_seconds=settings.error_alert_cooldown_seconds,
        fcm_configured=FCMClient(settings).configured,
    )


@router.post(
    "/alerts/test",
    response_model=AlertTestOut,
    status_code=status.HTTP_200_OK,
    summary="Send a test alert to administrators",
)
async def test_alert(request: Request, db: DB, settings: SettingsDep, admin: AdminUser) -> AlertTestOut:
    monitor = _monitor(request) or SystemMonitor(db, settings, role="api")
    result: dict[str, Any] = await monitor.notify_admins(
        "[Quantachain] Test alert",
        f"Test alert triggered by {admin.email}. Alerting pipeline is working.",
        severity="info",
    )
    return AlertTestOut(**result)


def _mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"
