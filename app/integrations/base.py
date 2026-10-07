"""Shared async HTTP client for third-party providers.

Adds what every integration needs: timeouts, bounded retries with backoff on
transient failures, rate-limit header parsing and per-provider usage metrics
that surface in the admin dashboard.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from app.core.exceptions import ExternalServiceError
from app.core.metrics import metrics

logger = structlog.get_logger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ProviderClient:
    """Thin wrapper around ``httpx.AsyncClient`` bound to one provider."""

    def __init__(
        self,
        name: str,
        base_url: str = "",
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
        retries: int = 2,
        params: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "User-Agent": "Quantachain/0.1 (+https://github.com/m-usman-k/quantachain-backend)",
                **(headers or {}),
            },
            timeout=timeout,
            params=params,
            follow_redirects=True,
        )
        self.stats = metrics.provider(name)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ProviderClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------ requests
    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                response = await self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                latency = (time.perf_counter() - started) * 1000
                self.stats.record(latency, error=f"{type(exc).__name__}: {exc}")
                if attempt >= self.retries:
                    raise ExternalServiceError(f"{self.name}: {type(exc).__name__}") from exc
                await self._backoff(attempt)
                attempt += 1
                continue

            latency = (time.perf_counter() - started) * 1000
            self._capture_rate_limit(response)
            if response.status_code in RETRYABLE_STATUS and attempt < self.retries:
                self.stats.record(latency, error=f"HTTP {response.status_code}")
                retry_after = response.headers.get("Retry-After")
                await self._backoff(attempt, retry_after)
                attempt += 1
                continue
            if response.is_error:
                self.stats.record(latency, error=f"HTTP {response.status_code}")
                raise ExternalServiceError(
                    f"{self.name} returned HTTP {response.status_code}",
                    details={"status": response.status_code, "body": response.text[:300]},
                )
            self.stats.record(latency)
            return response

    async def get_json(self, url: str, **kwargs: Any) -> Any:
        response = await self.request("GET", url, **kwargs)
        return _json(response, self.name)

    async def post_json(self, url: str, **kwargs: Any) -> Any:
        response = await self.request("POST", url, **kwargs)
        return _json(response, self.name)

    # ------------------------------------------------------------- helpers
    def _capture_rate_limit(self, response: httpx.Response) -> None:
        headers = response.headers
        remaining = headers.get("x-ratelimit-remaining") or headers.get("x-rate-limit-remaining")
        if remaining is not None and remaining.isdigit():
            self.stats.rate_limit_remaining = int(remaining)
        reset = headers.get("x-ratelimit-reset") or headers.get("x-rate-limit-reset")
        if reset is not None and reset.isdigit():
            value = int(reset)
            # Either an epoch timestamp or "seconds until reset".
            self.stats.rate_limit_reset_at = (
                datetime.fromtimestamp(value, tz=UTC)
                if value > 10_000_000
                else datetime.fromtimestamp(time.time() + value, tz=UTC)
            )

    async def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after and retry_after.isdigit():
            delay = min(float(retry_after), 30.0)
        else:
            delay = min(0.5 * (2**attempt) + random.random() * 0.25, 8.0)
        logger.debug("provider_retry", provider=self.name, attempt=attempt + 1, delay=round(delay, 2))
        await asyncio.sleep(delay)


def _json(response: httpx.Response, provider: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise ExternalServiceError(f"{provider} returned a non-JSON response") from exc


__all__ = ["RETRYABLE_STATUS", "ProviderClient"]
