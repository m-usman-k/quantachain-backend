"""Module 7 - admin monitoring, controls, user management, audit export."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.db.collections import Collections
from tests.conftest import API, bearer, register


async def test_admin_endpoints_require_admin_role(client, user_account) -> None:  # type: ignore[no-untyped-def]
    for path in (
        "/admin/overview",
        "/admin/health",
        "/admin/system/metrics",
        "/admin/jobs",
        "/admin/models",
        "/admin/users",
        "/admin/audit-logs",
    ):
        response = await client.get(f"{API}{path}", headers=user_account["headers"])
        assert response.status_code == 403, path


async def test_system_metrics_and_health(client, admin_account) -> None:  # type: ignore[no-untyped-def]
    metrics = await client.get(f"{API}/admin/system/metrics", headers=admin_account["headers"])
    assert metrics.status_code == 200
    body = metrics.json()
    assert 0 <= body["cpu_percent"] <= 100
    assert body["memory"]["percent"] > 0
    assert body["process"]["rss_bytes"] > 0

    health = await client.get(f"{API}/admin/health", headers=admin_account["headers"])
    assert health.status_code == 200
    data = health.json()
    assert data["database"]["connected"] is True
    assert data["api"]["requests_total"] > 0
    assert "users" in data["data_volume"]
    assert data["status"] in {"ok", "degraded"}

    overview = await client.get(f"{API}/admin/overview", headers=admin_account["headers"])
    assert overview.status_code == 200
    assert overview.json()["registered_users"] >= 1


async def test_metrics_history_after_sample(client, admin_account, db) -> None:  # type: ignore[no-untyped-def]
    from app.modules.admin.monitoring import SystemMonitor

    monitor = SystemMonitor(db, role="api")
    await monitor.sample()
    response = await client.get(
        f"{API}/admin/system/metrics/history",
        params={"range": "1h", "interval": "1m"},
        headers=admin_account["headers"],
    )
    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert response.json()[-1]["cpu_percent"] is not None


async def test_job_controls(client, admin_account, db) -> None:  # type: ignore[no-untyped-def]
    await db[Collections.SCHEDULER_JOBS].update_one(
        {"name": "demo_job"},
        {
            "$set": {
                "name": "demo_job",
                "module": "test",
                "interval_seconds": 60,
                "enabled": True,
                "paused": False,
                "running": False,
                "run_count": 3,
            }
        },
        upsert=True,
    )
    headers = admin_account["headers"]
    listing = await client.get(f"{API}/admin/jobs", headers=headers)
    assert listing.status_code == 200
    assert any(j["name"] == "demo_job" for j in listing.json())

    paused = await client.post(f"{API}/admin/jobs/demo_job/pause", headers=headers)
    assert paused.status_code == 200 and paused.json()["paused"] is True
    resumed = await client.post(f"{API}/admin/jobs/demo_job/resume", headers=headers)
    assert resumed.json()["paused"] is False
    run = await client.post(f"{API}/admin/jobs/demo_job/run", headers=headers)
    assert run.json()["run_requested_at"] is not None
    assert (await client.post(f"{API}/admin/jobs/missing/run", headers=headers)).status_code == 404


async def test_model_pause_and_override(client, admin_account, db) -> None:  # type: ignore[no-untyped-def]
    from app.modules.admin.controls import ModelControlService

    headers = admin_account["headers"]
    models = await client.get(f"{API}/admin/models", headers=headers)
    assert models.status_code == 200
    names = {m["name"] for m in models.json()}
    assert {"sentiment", "price_forecaster", "fraud_scanner", "whale_detector"} <= names

    paused = await client.post(f"{API}/admin/models/sentiment/pause", json={"reason": "maintenance"}, headers=headers)
    assert paused.status_code == 200
    assert paused.json()["paused"] is True and paused.json()["pause_reason"] == "maintenance"
    assert await ModelControlService(db).is_paused("sentiment") is True
    resumed = await client.post(f"{API}/admin/models/sentiment/resume", headers=headers)
    assert resumed.json()["paused"] is False

    override = await client.put(
        f"{API}/admin/models/sentiment/override",
        json={"override": {"index": 72}, "expires_in_minutes": 30},
        headers=headers,
    )
    assert override.status_code == 200
    assert override.json()["override_active"] is True
    assert await ModelControlService(db).active_override("sentiment") == {"index": 72}
    cleared = await client.delete(f"{API}/admin/models/sentiment/override", headers=headers)
    assert cleared.json()["override_active"] is False
    assert (await client.post(f"{API}/admin/models/nonexistent/pause", json={}, headers=headers)).status_code == 404


async def test_user_management(client, admin_account) -> None:  # type: ignore[no-untyped-def]
    headers = admin_account["headers"]
    target = await register(client)
    target_headers = bearer(target["tokens"])
    target_id = target["user"]["id"]

    listing = await client.get(f"{API}/admin/users", params={"q": target["user"]["email"][:12]}, headers=headers)
    assert listing.status_code == 200
    assert any(u["id"] == target_id for u in listing.json()["items"])

    promoted = await client.patch(f"{API}/admin/users/{target_id}", json={"role": "admin"}, headers=headers)
    assert promoted.status_code == 200 and promoted.json()["role"] == "admin"
    # Role change invalidates the user's existing access tokens.
    assert (await client.get(f"{API}/auth/me", headers=target_headers)).status_code == 401

    login = await client.post(
        f"{API}/auth/login", json={"email": target["user"]["email"], "password": target["password"]}
    )
    fresh_headers = bearer(login.json()["tokens"])
    deactivated = await client.patch(f"{API}/admin/users/{target_id}", json={"is_active": False}, headers=headers)
    assert deactivated.json()["is_active"] is False
    assert (await client.get(f"{API}/auth/me", headers=fresh_headers)).status_code == 401
    assert (
        await client.post(f"{API}/auth/login", json={"email": target["user"]["email"], "password": target["password"]})
    ).status_code == 403

    self_demote = await client.patch(
        f"{API}/admin/users/{admin_account['user']['id']}", json={"role": "user"}, headers=headers
    )
    assert self_demote.status_code == 422

    reset = await client.post(f"{API}/admin/users/{target_id}/reset-mfa", headers=headers)
    assert reset.status_code == 200 and reset.json()["mfa_enabled"] is False
    revoked = await client.post(f"{API}/admin/users/{target_id}/revoke-sessions", headers=headers)
    assert revoked.status_code == 200


async def test_audit_logs_filters_verify_and_export(client, admin_account) -> None:  # type: ignore[no-untyped-def]
    headers = admin_account["headers"]
    logs = await client.get(f"{API}/admin/audit-logs", params={"action": "user.", "limit": 5}, headers=headers)
    assert logs.status_code == 200
    assert logs.json()["total"] >= 1
    assert all(item["action"].startswith("user.") for item in logs.json()["items"])

    since = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    future = await client.get(f"{API}/admin/audit-logs", params={"since": since}, headers=headers)
    assert future.json()["total"] == 0

    verify = await client.get(f"{API}/admin/audit-logs/verify", headers=headers)
    assert verify.status_code == 200 and verify.json()["valid"] is True

    csv_export = await client.get(f"{API}/admin/audit-logs/export", params={"format": "csv"}, headers=headers)
    assert csv_export.status_code == 200
    assert csv_export.headers["content-type"].startswith("text/csv")
    lines = csv_export.text.strip().splitlines()
    assert lines[0].startswith("seq,created_at,actor_id")
    assert len(lines) >= 2

    json_export = await client.get(f"{API}/admin/audit-logs/export", params={"format": "json"}, headers=headers)
    assert json_export.status_code == 200
    assert isinstance(json_export.json(), list) and json_export.json()


async def test_logs_providers_and_alerts(client, admin_account) -> None:  # type: ignore[no-untyped-def]
    headers = admin_account["headers"]
    recent = await client.get(f"{API}/admin/logs/recent", params={"limit": 20}, headers=headers)
    assert recent.status_code == 200
    export = await client.get(f"{API}/admin/logs/export", params={"format": "json", "limit": 50}, headers=headers)
    assert export.status_code == 200
    providers = await client.get(f"{API}/admin/providers", headers=headers)
    assert providers.status_code == 200

    config = await client.get(f"{API}/admin/alerts/config", headers=headers)
    assert config.status_code == 200 and config.json()["smtp_configured"] is False

    test_alert = await client.post(f"{API}/admin/alerts/test", headers=headers)
    assert test_alert.status_code == 200
    assert test_alert.json()["email_sent"] is False  # SMTP not configured in tests
    assert test_alert.json()["admins_notified"] >= 1
    inbox = await client.get(f"{API}/notifications", params={"unread_only": True}, headers=headers)
    assert any(n["type"] == "system" for n in inbox.json()["items"])
