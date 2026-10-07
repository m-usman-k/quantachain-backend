"""Generic repository with typed CRUD on top of a Mongo collection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Generic, TypeVar

from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.core.exceptions import NotFoundError
from app.core.pagination import PageParams
from app.core.timeutils import utcnow
from app.db.models import Document, to_object_id

D = TypeVar("D", bound=Document)

SortSpec = Sequence[tuple[str, int]]


class Repository(Generic[D]):
    collection_name: ClassVar[str]
    model: ClassVar[type[Document]]
    not_found_message: ClassVar[str] = "Resource not found"

    def __init__(self, db: AsyncDatabase[dict[str, Any]]) -> None:
        self.db = db
        self.col: AsyncCollection[dict[str, Any]] = db[self.collection_name]

    # ------------------------------------------------------------- parsing
    def _parse(self, raw: dict[str, Any]) -> D:
        return self.model.model_validate(raw)  # type: ignore[return-value]

    # -------------------------------------------------------------- create
    async def insert(self, doc: D) -> D:
        result = await self.col.insert_one(doc.to_mongo())
        doc.id = str(result.inserted_id)
        return doc

    async def insert_many(self, docs: Sequence[D], *, ordered: bool = False) -> list[D]:
        if not docs:
            return []
        result = await self.col.insert_many([d.to_mongo() for d in docs], ordered=ordered)
        for doc, inserted_id in zip(docs, result.inserted_ids, strict=False):
            doc.id = str(inserted_id)
        return list(docs)

    # ---------------------------------------------------------------- read
    async def get(self, doc_id: str) -> D | None:
        raw = await self.col.find_one({"_id": to_object_id(doc_id)})
        return self._parse(raw) if raw else None

    async def get_or_raise(self, doc_id: str) -> D:
        doc = await self.get(doc_id)
        if doc is None:
            raise NotFoundError(self.not_found_message)
        return doc

    async def find_one(self, filter: dict[str, Any], *, sort: SortSpec | None = None) -> D | None:
        raw = await self.col.find_one(filter, sort=list(sort) if sort else None)
        return self._parse(raw) if raw else None

    async def find(
        self,
        filter: dict[str, Any] | None = None,
        *,
        sort: SortSpec | None = None,
        limit: int = 0,
        skip: int = 0,
        projection: dict[str, Any] | None = None,
    ) -> list[D]:
        cursor = self.col.find(filter or {}, projection)
        if sort:
            cursor = cursor.sort(list(sort))
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return [self._parse(raw) async for raw in cursor]

    async def count(self, filter: dict[str, Any] | None = None) -> int:
        return await self.col.count_documents(filter or {})

    async def exists(self, filter: dict[str, Any]) -> bool:
        return await self.col.find_one(filter, {"_id": 1}) is not None

    async def paginate(
        self,
        filter: dict[str, Any] | None,
        params: PageParams,
        *,
        sort: SortSpec | None = None,
    ) -> tuple[list[D], int]:
        items = await self.find(filter, sort=sort, limit=params.limit, skip=params.offset)
        total = await self.count(filter)
        return items, total

    # -------------------------------------------------------------- update
    async def update(
        self,
        doc_id: str,
        set_fields: dict[str, Any] | None = None,
        *,
        unset: Sequence[str] | None = None,
        inc: dict[str, int | float] | None = None,
        push: dict[str, Any] | None = None,
        pull: dict[str, Any] | None = None,
        add_to_set: dict[str, Any] | None = None,
    ) -> D | None:
        update = self._build_update(set_fields, unset=unset, inc=inc, push=push, pull=pull, add_to_set=add_to_set)
        raw = await self.col.find_one_and_update(
            {"_id": to_object_id(doc_id)}, update, return_document=ReturnDocument.AFTER
        )
        return self._parse(raw) if raw else None

    async def update_where(self, filter: dict[str, Any], set_fields: dict[str, Any], **kwargs: Any) -> D | None:
        raw = await self.col.find_one_and_update(
            filter, self._build_update(set_fields, **kwargs), return_document=ReturnDocument.AFTER
        )
        return self._parse(raw) if raw else None

    async def update_many(self, filter: dict[str, Any], set_fields: dict[str, Any], **kwargs: Any) -> int:
        result = await self.col.update_many(filter, self._build_update(set_fields, **kwargs))
        return result.modified_count

    async def upsert(
        self, filter: dict[str, Any], set_fields: dict[str, Any], *, on_insert: dict[str, Any] | None = None
    ) -> D:
        update = self._build_update(set_fields)
        insert_defaults = {"created_at": utcnow()}
        if on_insert:
            insert_defaults.update(on_insert)
        # Fields in $set must not also appear in $setOnInsert.
        update["$setOnInsert"] = {k: v for k, v in insert_defaults.items() if k not in update["$set"]}
        raw = await self.col.find_one_and_update(filter, update, upsert=True, return_document=ReturnDocument.AFTER)
        return self._parse(raw)

    async def replace(self, doc: D) -> D:
        if doc.id is None:
            raise ValueError("Cannot replace a document without an id")
        doc.updated_at = utcnow()
        await self.col.replace_one({"_id": to_object_id(doc.id)}, doc.to_mongo(), upsert=True)
        return doc

    # -------------------------------------------------------------- delete
    async def delete(self, doc_id: str) -> bool:
        result = await self.col.delete_one({"_id": to_object_id(doc_id)})
        return result.deleted_count == 1

    async def delete_many(self, filter: dict[str, Any]) -> int:
        result = await self.col.delete_many(filter)
        return result.deleted_count

    # ----------------------------------------------------------- aggregate
    async def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cursor = await self.col.aggregate(pipeline)
        return await cursor.to_list(length=None)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _build_update(
        set_fields: dict[str, Any] | None,
        *,
        unset: Sequence[str] | None = None,
        inc: dict[str, int | float] | None = None,
        push: dict[str, Any] | None = None,
        pull: dict[str, Any] | None = None,
        add_to_set: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        update: dict[str, Any] = {"$set": {**(set_fields or {}), "updated_at": utcnow()}}
        if unset:
            update["$unset"] = {field: "" for field in unset}
        if inc:
            update["$inc"] = inc
        if push:
            update["$push"] = push
        if pull:
            update["$pull"] = pull
        if add_to_set:
            update["$addToSet"] = add_to_set
        return update


__all__ = ["Repository", "SortSpec"]
