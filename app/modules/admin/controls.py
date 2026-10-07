"""Administrative pause / override switches for AI models and analytical pipelines (Module 7, FE-2).

Every long-running analytical job calls ``await ModelControlService(db).is_paused(name)``
before doing work. Admins flip the switch through ``POST /admin/models/{name}/pause``.
Overrides let an admin pin a manual value (for example a sentiment index or a
prediction) that services return instead of the model output while active.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field
from pymongo.asynchronous.database import AsyncDatabase

from app.core.exceptions import NotFoundError
from app.core.timeutils import utcnow
from app.db.collections import Collections
from app.db.models import Document
from app.db.repository import Repository


class ModelControl(Document):
    name: str  # e.g. "sentiment", "lstm_forecaster", "fraud_scanner", "whale_detector"
    module: str
    description: str = ""
    paused: bool = False
    paused_by: str | None = None
    paused_at: datetime | None = None
    pause_reason: str | None = None
    override: dict[str, Any] | None = None  # manual value used instead of model output
    override_by: str | None = None
    override_expires_at: datetime | None = None
    last_run_at: datetime | None = None
    last_run_status: str | None = None
    runs: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def override_active(self) -> bool:
        if not self.override:
            return False
        return self.override_expires_at is None or self.override_expires_at > utcnow()


# Models known to the platform; others can be registered dynamically by services.
KNOWN_MODELS: dict[str, tuple[str, str]] = {
    "whale_detector": ("onchain", "Block scanning and whale transfer detection"),
    "wallet_scorer": ("onchain", "Smart-money confidence scoring"),
    "sentiment": ("sentiment", "GPT-4o news / social sentiment scoring"),
    "geopolitical": ("sentiment", "Geopolitical impact modelling"),
    "price_forecaster": ("prediction", "LSTM / statistical price forecasting"),
    "volatility": ("prediction", "Volatility regime forecasting"),
    "signal_engine": ("prediction", "Automated trade signal emission"),
    "fraud_scanner": ("fraud", "Smart-contract vulnerability scanning"),
    "liquidity_monitor": ("fraud", "Rug-pull / pump-and-dump detection"),
    "research_assistant": ("research", "AI research chat assistant"),
    "insights": ("research", "Monthly / quarterly macro insights"),
}


class ModelControlRepository(Repository[ModelControl]):
    collection_name = Collections.MODEL_CONTROLS
    model = ModelControl
    not_found_message = "Model not found"


class ModelControlService:
    def __init__(self, db: AsyncDatabase[dict[str, Any]]) -> None:
        self.repo = ModelControlRepository(db)

    async def ensure_known(self) -> None:
        for name, (module, description) in KNOWN_MODELS.items():
            await self.repo.col.update_one(
                {"name": name},
                {
                    "$setOnInsert": {
                        "name": name,
                        "module": module,
                        "description": description,
                        "paused": False,
                        "runs": 0,
                        "created_at": utcnow(),
                    }
                },
                upsert=True,
            )

    async def get(self, name: str) -> ModelControl:
        control = await self.repo.find_one({"name": name})
        if control is None:
            if name not in KNOWN_MODELS:
                raise NotFoundError(f"Unknown model '{name}'")
            module, description = KNOWN_MODELS[name]
            control = await self.repo.upsert({"name": name}, {"module": module, "description": description})
        return control

    async def list(self) -> list[ModelControl]:
        await self.ensure_known()
        return await self.repo.find({}, sort=[("module", 1), ("name", 1)])

    async def is_paused(self, name: str) -> bool:
        doc = await self.repo.col.find_one({"name": name}, {"paused": 1})
        return bool(doc and doc.get("paused"))

    async def active_override(self, name: str) -> dict[str, Any] | None:
        control = await self.repo.find_one({"name": name})
        return control.override if control and control.override_active else None

    async def set_paused(
        self, name: str, paused: bool, *, actor: str | None, reason: str | None = None
    ) -> ModelControl:
        await self.get(name)
        fields: dict[str, Any] = {"paused": paused}
        fields.update(
            {"paused_by": actor, "paused_at": utcnow(), "pause_reason": reason}
            if paused
            else {"paused_by": None, "paused_at": None, "pause_reason": None}
        )
        return await self.repo.update_where({"name": name}, fields)  # type: ignore[return-value]

    async def set_override(
        self, name: str, override: dict[str, Any] | None, *, actor: str | None, expires_at: datetime | None = None
    ) -> ModelControl:
        await self.get(name)
        return await self.repo.update_where(  # type: ignore[return-value]
            {"name": name},
            {
                "override": override,
                "override_by": actor if override else None,
                "override_expires_at": expires_at if override else None,
            },
        )

    async def record_run(self, name: str, *, status: str = "ok", **extra: Any) -> None:
        await self.repo.col.update_one(
            {"name": name},
            {
                "$set": {"last_run_at": utcnow(), "last_run_status": status, **({"extra": extra} if extra else {})},
                "$inc": {"runs": 1},
            },
            upsert=True,
        )


__all__ = ["KNOWN_MODELS", "ModelControl", "ModelControlRepository", "ModelControlService"]
