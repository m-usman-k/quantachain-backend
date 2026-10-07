"""Base document model and BSON-friendly field types."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from bson import ObjectId
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from app.core.timeutils import utcnow


def _stringify_id(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    return value


ObjectIdStr = Annotated[str, BeforeValidator(_stringify_id)]


def to_object_id(value: str | ObjectId) -> ObjectId | str:
    """Convert a 24-hex string to ObjectId; other strings are used as-is (custom ids)."""
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and len(value) == 24 and ObjectId.is_valid(value):
        return ObjectId(value)
    return value


class Document(BaseModel):
    """Base class for everything persisted in MongoDB.

    ``id`` maps to Mongo's ``_id``; it is ``None`` until the document is inserted.
    """

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True, extra="ignore")

    id: ObjectIdStr | None = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime | None = None

    def to_mongo(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True)
        if data.get("_id") is None:
            data.pop("_id", None)
        else:
            data["_id"] = to_object_id(data["_id"])
        return data

    @classmethod
    def from_mongo(cls, raw: dict[str, Any]):  # type: ignore[no-untyped-def]
        return cls.model_validate(raw)


__all__ = ["Document", "ObjectIdStr", "to_object_id"]
