"""Cross-process live log stream backed by a capped collection.

Workers write one line per noteworthy ingestion event; the API tails the
collection to feed ``/ws/ingestion`` (the "Live Ingestion Stream" panel).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from pymongo.asynchronous.database import AsyncDatabase

from app.core.events import Topics, event_bus
from app.core.timeutils import utcnow
from app.db.collections import Collections


async def append_ingestion_log(
    db: AsyncDatabase[dict[str, Any]],
    source: str,
    message: str,
    *,
    level: str = "info",
    **fields: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"ts": utcnow(), "source": source, "level": level, "message": message, **fields}
    result = await db[Collections.INGESTION_LOGS].insert_one(entry)
    entry["_id"] = result.inserted_id
    event_bus.publish(Topics.INGESTION_LOG, serialise_log(entry))
    return entry


async def tail_ingestion_logs(
    db: AsyncDatabase[dict[str, Any]], *, after_id: ObjectId | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"_id": {"$gt": after_id}} if after_id else {}
    cursor = db[Collections.INGESTION_LOGS].find(query).sort("_id", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    rows.reverse()
    return rows


def serialise_log(entry: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in entry.items() if k != "_id"}
    out["id"] = str(entry.get("_id", ""))
    ts = out.get("ts")
    if isinstance(ts, datetime):
        out["ts"] = ts.isoformat()
    return out


__all__ = ["append_ingestion_log", "serialise_log", "tail_ingestion_logs"]
