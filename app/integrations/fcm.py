"""Firebase Cloud Messaging (HTTP v1) client without the firebase-admin SDK.

Authenticates with a service-account JSON (``FCM_SERVICE_ACCOUNT_JSON`` holds
either the JSON text or a path to it) by signing an RS256 JWT and exchanging it
for an OAuth2 access token, then posts one message per device token.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import jwt
import structlog

from app.core.config import Settings, get_settings
from app.core.metrics import metrics

logger = structlog.get_logger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
INVALID_TOKEN_ERRORS = {"UNREGISTERED", "INVALID_ARGUMENT", "NOT_FOUND"}


@dataclass
class PushResult:
    sent: int = 0
    failed: int = 0
    skipped: bool = False
    invalid_tokens: list[str] = field(default_factory=list)


class FCMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._service_account: dict[str, Any] | None = None
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        self.stats = metrics.provider("fcm")

    @property
    def configured(self) -> bool:
        return bool(self.settings.fcm_service_account_json) and bool(
            self.settings.fcm_project_id or self._load().get("project_id")
        )

    def _load(self) -> dict[str, Any]:
        if self._service_account is not None:
            return self._service_account
        raw = (
            self.settings.fcm_service_account_json.get_secret_value() if self.settings.fcm_service_account_json else ""
        )
        if not raw:
            self._service_account = {}
            return self._service_account
        text = raw.strip()
        if not text.startswith("{") and Path(text).exists():
            text = Path(text).read_text(encoding="utf-8")
        try:
            self._service_account = json.loads(text)
        except ValueError:
            logger.error("fcm_service_account_invalid")
            self._service_account = {}
        return self._service_account

    async def _get_access_token(self, client: httpx.AsyncClient) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        account = self._load()
        now = int(time.time())
        assertion = jwt.encode(
            {
                "iss": account["client_email"],
                "scope": FCM_SCOPE,
                "aud": TOKEN_URL,
                "iat": now,
                "exp": now + 3600,
            },
            account["private_key"],
            algorithm="RS256",
        )
        response = await client.post(
            TOKEN_URL,
            data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": assertion},
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = payload["access_token"]
        self._token_expires_at = time.time() + int(payload.get("expires_in", 3600))
        return self._access_token or ""

    async def send(
        self,
        tokens: list[str],
        *,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
    ) -> PushResult:
        result = PushResult()
        if not tokens:
            return result
        if not self.configured:
            result.skipped = True
            logger.debug("fcm_not_configured_skipping_push", recipients=len(tokens))
            return result
        project_id = self.settings.fcm_project_id or self._load().get("project_id")
        url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
        string_data = {k: str(v) for k, v in (data or {}).items()}
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                access_token = await self._get_access_token(client)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.error("fcm_auth_failed", error=str(exc))
                result.failed = len(tokens)
                self.stats.record(0.0, error=f"auth: {exc}")
                return result
            headers = {"Authorization": f"Bearer {access_token}"}
            for token in tokens:
                message = {
                    "message": {
                        "token": token,
                        "notification": {"title": title[:200], "body": body[:1000]},
                        "data": string_data,
                        "android": {"priority": "high"},
                        "apns": {"headers": {"apns-priority": "10"}},
                    }
                }
                started = time.perf_counter()
                try:
                    response = await client.post(url, json=message, headers=headers)
                except httpx.HTTPError as exc:
                    result.failed += 1
                    self.stats.record((time.perf_counter() - started) * 1000, error=str(exc))
                    continue
                latency = (time.perf_counter() - started) * 1000
                if response.is_success:
                    result.sent += 1
                    self.stats.record(latency)
                    continue
                result.failed += 1
                error_code = ""
                try:
                    details = response.json().get("error", {})
                    error_code = details.get("status", "")
                    for item in details.get("details", []):
                        error_code = item.get("errorCode", error_code)
                except ValueError:
                    pass
                self.stats.record(latency, error=f"HTTP {response.status_code} {error_code}")
                if error_code in INVALID_TOKEN_ERRORS:
                    result.invalid_tokens.append(token)
        return result


__all__ = ["FCMClient", "PushResult"]
