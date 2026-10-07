"""MongoDB connection lifecycle (PyMongo's native asyncio client)."""

from __future__ import annotations

import time
from typing import Any

import structlog
from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import PyMongoError

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError

logger = structlog.get_logger(__name__)


class MongoManager:
    def __init__(self) -> None:
        self._client: AsyncMongoClient[dict[str, Any]] | None = None
        self._db: AsyncDatabase[dict[str, Any]] | None = None

    async def connect(self, settings: Settings | None = None) -> AsyncDatabase[dict[str, Any]]:
        settings = settings or get_settings()
        if self._db is not None:
            return self._db
        self._client = AsyncMongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
            connectTimeoutMS=settings.mongodb_timeout_ms,
            tz_aware=True,
            uuidRepresentation="standard",
            appname=settings.app_name,
        )
        self._db = self._client[settings.mongodb_db]
        await self._client.admin.command("ping")
        logger.info("mongodb_connected", database=settings.mongodb_db)
        return self._db

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            logger.info("mongodb_closed")
        self._client = None
        self._db = None

    @property
    def connected(self) -> bool:
        return self._db is not None

    @property
    def client(self) -> AsyncMongoClient[dict[str, Any]]:
        if self._client is None:
            raise ServiceUnavailableError("Database connection is not initialised")
        return self._client

    @property
    def db(self) -> AsyncDatabase[dict[str, Any]]:
        if self._db is None:
            raise ServiceUnavailableError("Database connection is not initialised")
        return self._db

    async def ping_ms(self) -> float | None:
        try:
            started = time.perf_counter()
            await self.client.admin.command("ping")
            return round((time.perf_counter() - started) * 1000, 2)
        except (PyMongoError, ServiceUnavailableError):
            return None

    async def stats(self) -> dict[str, Any]:
        """Database-level storage statistics for the admin health dashboard."""
        raw = await self.db.command("dbStats")
        return {
            "database": raw.get("db"),
            "collections": raw.get("collections"),
            "objects": raw.get("objects"),
            "data_size_bytes": raw.get("dataSize"),
            "storage_size_bytes": raw.get("storageSize"),
            "index_size_bytes": raw.get("indexSize"),
            "total_size_bytes": raw.get("totalSize"),
        }

    async def server_version(self) -> str | None:
        try:
            info = await self.client.server_info()
            return str(info.get("version"))
        except (PyMongoError, ServiceUnavailableError):
            return None


mongo = MongoManager()


def get_db() -> AsyncDatabase[dict[str, Any]]:
    """FastAPI dependency returning the application database."""
    return mongo.db


__all__ = ["AsyncDatabase", "MongoManager", "get_db", "mongo"]
