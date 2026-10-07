"""Aggregates every module router under the versioned API prefix."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.health import router as health_router
from app.modules.auth.router import router as auth_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(auth_router)


def include_optional_routers() -> None:
    """Attach module routers as they land; missing modules never break start-up."""
    from importlib import import_module

    for module_path in (
        "app.modules.market.router",
        "app.modules.onchain.router",
        "app.modules.sentiment.router",
        "app.modules.prediction.router",
        "app.modules.fraud.router",
        "app.modules.admin.router",
        "app.modules.research.router",
        "app.modules.trading.router",
        "app.modules.notifications.router",
    ):
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name and module_path.startswith(exc.name):
                continue
            raise
        api_router.include_router(module.router)


include_optional_routers()

__all__ = ["api_router"]
