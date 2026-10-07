"""Small in-memory sliding-window rate limiter.

Good enough for a single API node; put a shared store (Redis) behind the same
interface when scaling horizontally. Disabled when ``RATE_LIMIT_PER_MINUTE=0``
or in the ``test`` environment.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request

from app.core.config import get_settings
from app.core.exceptions import RateLimitedError


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = time.monotonic()

    def check(self, key: str) -> tuple[bool, int, float]:
        """Return ``(allowed, remaining, retry_after_seconds)``."""
        now = time.monotonic()
        bucket = self._hits.setdefault(key, deque())
        cutoff = now - self.window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.limit:
            retry_after = max(0.0, bucket[0] + self.window - now)
            return False, 0, retry_after
        bucket.append(now)
        self._maybe_sweep(now)
        return True, self.limit - len(bucket), 0.0

    def _maybe_sweep(self, now: float) -> None:
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        cutoff = now - self.window
        for key in [k for k, dq in self._hits.items() if not dq or dq[-1] <= cutoff]:
            del self._hits[key]

    def reset(self) -> None:
        self._hits.clear()


def client_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def rate_limited(
    limit: int | None = None, window_seconds: float = 60.0
) -> Callable[[Request], Coroutine[Any, Any, None]]:
    """Dependency factory: ``Depends(rate_limited(10))`` allows 10 calls/minute per client."""
    limiter_holder: dict[str, SlidingWindowLimiter] = {}

    async def dependency(request: Request) -> None:
        settings = get_settings()
        effective = settings.rate_limit_per_minute if limit is None else limit
        if effective <= 0 or settings.is_test:
            return
        limiter = limiter_holder.get("limiter")
        if limiter is None or limiter.limit != effective:
            limiter = SlidingWindowLimiter(effective, window_seconds)
            limiter_holder["limiter"] = limiter
        allowed, _remaining, retry_after = limiter.check(client_key(request))
        if not allowed:
            raise RateLimitedError(
                "Rate limit exceeded, slow down",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    return dependency


__all__ = ["SlidingWindowLimiter", "client_key", "rate_limited"]
