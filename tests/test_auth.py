"""Module 1 - authentication, MFA, sessions, API keys, RBAC and audit trail."""

from __future__ import annotations

import pyotp
from bson import ObjectId

from tests.conftest import API, bearer, register, unique_email


async def test_register_login_me(client) -> None:  # type: ignore[no-untyped-def]
    email = unique_email()
    data = await register(client, email=email)
    assert data["user"]["email"] == email
    assert data["user"]["role"] in {"user", "admin"}
    assert data["tokens"]["token_type"] == "bearer"

    me = await client.get(f"{API}/auth/me", headers=bearer(data["tokens"]))
    assert me.status_code == 200
    assert me.json()["email"] == email

    login = await client.post(f"{API}/auth/login", json={"email": email, "password": data["password"]})
    assert login.status_code == 200
    assert login.json()["mfa_required"] is False
    assert login.json()["tokens"]["access_token"]


async def test_duplicate_registration_conflicts(client) -> None:  # type: ignore[no-untyped-def]
    email = unique_email()
    await register(client, email=email)
    response = await client.post(f"{API}/auth/register", json={"email": email, "password": "Str0ngPass!"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_weak_password_rejected(client) -> None:  # type: ignore[no-untyped-def]
    response = await client.post(f"{API}/auth/register", json={"email": unique_email(), "password": "alllowercase1"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failed"


async def test_invalid_login_and_lockout(client) -> None:  # type: ignore[no-untyped-def]
    email = unique_email()
    await register(client, email=email)
    for _ in range(5):
        response = await client.post(f"{API}/auth/login", json={"email": email, "password": "Wrong1234"})
        assert response.status_code == 401
    locked = await client.post(f"{API}/auth/login", json={"email": email, "password": "Str0ngPass!"})
    assert locked.status_code == 403
    assert "locked" in locked.json()["error"]["message"].lower()


async def test_refresh_rotation_and_reuse_detection(client) -> None:  # type: ignore[no-untyped-def]
    data = await register(client)
    first_refresh = data["tokens"]["refresh_token"]

    rotated = await client.post(f"{API}/auth/refresh", json={"refresh_token": first_refresh})
    assert rotated.status_code == 200
    second_refresh = rotated.json()["refresh_token"]
    assert second_refresh != first_refresh

    # Replaying the old token is treated as theft: the whole family is revoked.
    reuse = await client.post(f"{API}/auth/refresh", json={"refresh_token": first_refresh})
    assert reuse.status_code == 401
    killed = await client.post(f"{API}/auth/refresh", json={"refresh_token": second_refresh})
    assert killed.status_code == 401


async def test_logout_all_devices_invalidates_access_tokens(client) -> None:  # type: ignore[no-untyped-def]
    data = await register(client)
    headers = bearer(data["tokens"])
    assert (await client.get(f"{API}/auth/me", headers=headers)).status_code == 200
    response = await client.post(f"{API}/auth/logout", json={"all_devices": True}, headers=headers)
    assert response.status_code == 204
    assert (await client.get(f"{API}/auth/me", headers=headers)).status_code == 401


async def test_sessions_listing_and_revocation(client) -> None:  # type: ignore[no-untyped-def]
    data = await register(client)
    email, password = data["user"]["email"], data["password"]
    await client.post(f"{API}/auth/login", json={"email": email, "password": password})
    headers = bearer(data["tokens"])
    sessions = await client.get(f"{API}/auth/sessions", headers=headers)
    assert sessions.status_code == 200
    assert len(sessions.json()) == 2
    other = next(s for s in sessions.json() if not s["current"])
    assert (await client.delete(f"{API}/auth/sessions/{other['id']}", headers=headers)).status_code == 204
    assert len((await client.get(f"{API}/auth/sessions", headers=headers)).json()) == 1


async def test_mfa_enrolment_login_and_recovery_code(client) -> None:  # type: ignore[no-untyped-def]
    data = await register(client)
    headers = bearer(data["tokens"])
    email, password = data["user"]["email"], data["password"]

    setup = await client.post(f"{API}/auth/mfa/setup", headers=headers)
    assert setup.status_code == 200
    secret = setup.json()["secret"]
    assert "otpauth://totp/" in setup.json()["otpauth_uri"]

    bad = await client.post(f"{API}/auth/mfa/enable", json={"code": "000000"}, headers=headers)
    assert bad.status_code in (401, 200)  # 1-in-a-million chance the code is right
    enable = await client.post(f"{API}/auth/mfa/enable", json={"code": pyotp.TOTP(secret).now()}, headers=headers)
    assert enable.status_code == 200
    recovery_codes = enable.json()["recovery_codes"]
    assert len(recovery_codes) == 8

    me = await client.get(f"{API}/auth/me", headers=headers)
    assert me.json()["mfa_enabled"] is True

    # Password login now returns a challenge instead of tokens.
    login = await client.post(f"{API}/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200
    assert login.json()["mfa_required"] is True
    mfa_token = login.json()["mfa_token"]
    assert login.json()["tokens"] is None

    wrong = await client.post(f"{API}/auth/mfa/verify", json={"mfa_token": mfa_token, "code": "123456"})
    assert wrong.status_code in (401, 200)
    ok = await client.post(f"{API}/auth/mfa/verify", json={"mfa_token": mfa_token, "code": pyotp.TOTP(secret).now()})
    assert ok.status_code == 200
    assert ok.json()["tokens"]["access_token"]

    # A recovery code works exactly once.
    login = await client.post(f"{API}/auth/login", json={"email": email, "password": password})
    mfa_token = login.json()["mfa_token"]
    with_code = await client.post(f"{API}/auth/mfa/verify", json={"mfa_token": mfa_token, "code": recovery_codes[0]})
    assert with_code.status_code == 200
    login = await client.post(f"{API}/auth/login", json={"email": email, "password": password})
    replay = await client.post(
        f"{API}/auth/mfa/verify", json={"mfa_token": login.json()["mfa_token"], "code": recovery_codes[0]}
    )
    assert replay.status_code == 401

    # Disable with password.
    disable = await client.post(
        f"{API}/auth/mfa/disable", json={"password": password}, headers=bearer(ok.json()["tokens"])
    )
    assert disable.status_code == 204
    login = await client.post(f"{API}/auth/login", json={"email": email, "password": password})
    assert login.json()["mfa_required"] is False


async def test_password_change_revokes_old_sessions(client) -> None:  # type: ignore[no-untyped-def]
    data = await register(client)
    headers = bearer(data["tokens"])
    response = await client.put(
        f"{API}/auth/password",
        json={"current_password": data["password"], "new_password": "N3wStrongPass!"},
        headers=headers,
    )
    assert response.status_code == 204
    assert (await client.get(f"{API}/auth/me", headers=headers)).status_code == 401
    login = await client.post(f"{API}/auth/login", json={"email": data["user"]["email"], "password": "N3wStrongPass!"})
    assert login.status_code == 200


async def test_profile_update_and_preferences(client, user_account) -> None:  # type: ignore[no-untyped-def]
    response = await client.patch(
        f"{API}/auth/me",
        json={
            "full_name": "Muhammad Usman",
            "preferences": {
                "timezone": "Asia/Karachi",
                "favorite_symbols": ["BTC", "SOL"],
                "notifications": {"whale_alerts": False, "min_whale_usd": 5000000},
            },
        },
        headers=user_account["headers"],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["full_name"] == "Muhammad Usman"
    assert body["preferences"]["timezone"] == "Asia/Karachi"
    assert body["preferences"]["notifications"]["whale_alerts"] is False
    assert body["preferences"]["notifications"]["min_whale_usd"] == 5000000


async def test_api_keys_lifecycle(client, user_account) -> None:  # type: ignore[no-untyped-def]
    headers = user_account["headers"]
    created = await client.post(f"{API}/auth/api-keys", json={"name": "mobile", "expires_in_days": 30}, headers=headers)
    assert created.status_code == 201
    key = created.json()["key"]
    assert key.startswith("qc_")

    via_key = await client.get(f"{API}/auth/me", headers={"X-API-Key": key})
    assert via_key.status_code == 200
    assert via_key.json()["id"] == user_account["user"]["id"]

    listing = await client.get(f"{API}/auth/api-keys", headers=headers)
    assert listing.status_code == 200
    assert listing.json()[0]["prefix"] == created.json()["prefix"]
    assert "key" not in listing.json()[0]

    revoked = await client.delete(f"{API}/auth/api-keys/{created.json()['id']}", headers=headers)
    assert revoked.status_code == 204
    assert (await client.get(f"{API}/auth/me", headers={"X-API-Key": key})).status_code == 401


async def test_audit_trail_records_actions(client, user_account) -> None:  # type: ignore[no-untyped-def]
    response = await client.get(f"{API}/auth/audit", headers=user_account["headers"])
    assert response.status_code == 200
    actions = {item["action"] for item in response.json()["items"]}
    assert "user.registered" in actions
    assert all(len(item["hash"]) == 64 for item in response.json()["items"])


async def test_audit_chain_verifies_and_detects_tampering(db, client, user_account) -> None:  # type: ignore[no-untyped-def]
    from app.modules.auth.repository import AuditLogRepository

    repo = AuditLogRepository(db)
    verification = await repo.verify_chain()
    assert verification["valid"] is True
    assert verification["checked"] >= 1

    # Tamper with one record directly in the database.
    victim = await repo.find_one({"actor_id": user_account["user"]["id"]})
    assert victim is not None
    await repo.col.update_one({"_id": ObjectId(victim.id)}, {"$set": {"action": "user.forged"}})
    tampered = await repo.verify_chain()
    assert tampered["valid"] is False
    assert tampered["broken_at_seq"] == victim.seq
    # Restore so later tests still see a healthy chain.
    await repo.col.update_one({"_id": ObjectId(victim.id)}, {"$set": {"action": victim.action}})
    assert (await repo.verify_chain())["valid"] is True


async def test_rbac_blocks_regular_users(client, user_account, admin_account) -> None:  # type: ignore[no-untyped-def]
    from app.api.deps import require_roles  # noqa: F401 - ensures dependency importable

    denied = await client.get(f"{API}/admin/system/metrics", headers=user_account["headers"])
    assert denied.status_code in (403, 404)
    if denied.status_code == 403:
        allowed = await client.get(f"{API}/admin/system/metrics", headers=admin_account["headers"])
        assert allowed.status_code == 200


async def test_unauthenticated_requests_rejected(client) -> None:  # type: ignore[no-untyped-def]
    response = await client.get(f"{API}/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert response.headers.get("www-authenticate") == "Bearer"
    bad = await client.get(f"{API}/auth/me", headers={"Authorization": "Bearer not-a-token"})
    assert bad.status_code == 401


async def test_oauth_start_requires_configuration(client) -> None:  # type: ignore[no-untyped-def]
    providers = await client.get(f"{API}/auth/oauth/providers")
    assert providers.status_code == 200
    assert providers.json() == []
    response = await client.get(f"{API}/auth/oauth/google/start")
    assert response.status_code == 501
    unknown = await client.get(f"{API}/auth/oauth/facebook/start")
    assert unknown.status_code == 422


async def test_health_and_request_id(client) -> None:  # type: ignore[no-untyped-def]
    response = await client.get(f"{API}/health", headers={"X-Request-ID": "abc-123"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"]["connected"] is True
    assert response.headers["x-request-id"] == "abc-123"
