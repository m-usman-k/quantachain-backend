"""Request context middleware: request IDs, timing, access logs and error counting."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.metrics import metrics

logger = structlog.get_logger("http")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record_request(_route_template(request), 500, latency_ms)
            logger.exception(
                "request_failed", method=request.method, path=request.url.path, latency_ms=round(latency_ms, 2)
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        metrics.record_request(_route_template(request), response.status_code, latency_ms)

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.1f}"

        if not request.url.path.endswith(("/health", "/health/live")):
            log = logger.warning if response.status_code >= 500 else logger.info
            log(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=round(latency_ms, 2),
                user_id=getattr(request.state, "user_id", None),
            )
        return response


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return f"{request.method} {path or request.url.path}"


__all__ = ["REQUEST_ID_HEADER", "RequestContextMiddleware"]
