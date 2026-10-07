"""Test harness.

Integration tests run against a real MongoDB:

* ``MONGODB_TEST_URI`` if set (CI uses a ``mongo:7`` service container), else
* an embedded ``mongod`` started by ``pymongo_inmemory`` (downloaded once).

Everything runs on a single session-scoped event loop so the async Mongo client
created by the application lifespan can be shared by all tests.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

TEST_SECRET = "test-secret-key-for-pytest-only-0123456789"


@pytest.fixture(scope="session")
def mongo_uri() -> Iterator[str]:
    explicit = os.environ.get("MONGODB_TEST_URI")
    if explicit:
        yield explicit
        return
    os.environ.setdefault("PYMONGOIM__MONGO_VERSION", "7.0")
    from pymongo_inmemory import Mongod
    from pymongo_inmemory.context import Context

    with Mongod(Context()) as mongod:
        yield mongod.connection_string


@pytest.fixture(scope="session")
def test_env(mongo_uri: str) -> Iterator[dict[str, str]]:
    db_name = f"quantachain_test_{uuid.uuid4().hex[:8]}"
    values = {
        "ENVIRONMENT": "test",
        "MONGODB_URI": mongo_uri,
        "MONGODB_DB": db_name,
        "SECRET_KEY": TEST_SECRET,
        "RUN_WORKERS_IN_API": "false",
        "WORKERS_ENABLED": "false",
        "BACKFILL_ON_START": "false",
        "RATE_LIMIT_PER_MINUTE": "0",
        "LOG_LEVEL": "WARNING",
        "DATA_DIR": os.path.join(os.getcwd(), "var", "test"),
        "FIRST_ADMIN_EMAIL": "",
        "OPENAI_API_KEY": "",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    from app.core.config import reset_settings_cache

    reset_settings_cache()
    yield values
    for key, old in previous.items():
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
    reset_settings_cache()


@pytest.fixture(scope="session")
async def app(test_env: dict[str, str]) -> AsyncIterator[FastAPI]:
    from app.core.config import get_settings
    from app.main import create_app

    settings = get_settings()
    application = create_app(settings)
    async with LifespanManager(application, startup_timeout=60, shutdown_timeout=30):
        yield application
    # Drop the per-run database; the client is closed by the lifespan.
    from pymongo import AsyncMongoClient

    from app.db.mongo import mongo

    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(test_env["MONGODB_URI"])
    await client.drop_database(test_env["MONGODB_DB"])
    await client.close()
    assert not mongo.connected


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
        yield http


@pytest.fixture(scope="session")
async def db(app: FastAPI):  # type: ignore[no-untyped-def]
    from app.db.mongo import mongo

    return mongo.db


# --------------------------------------------------------------- auth helpers
API = "/api/v1"


def unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@example.com"


async def register(
    client: httpx.AsyncClient, *, email: str | None = None, password: str = "Str0ngPass!"
) -> dict[str, Any]:
    response = await client.post(
        f"{API}/auth/register",
        json={"email": email or unique_email(), "password": password, "full_name": "Test User"},
    )
    assert response.status_code == 201, response.text
    data = response.json()
    data["password"] = password
    return data


def bearer(tokens: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.fixture
async def user_account(client: httpx.AsyncClient) -> dict[str, Any]:
    data = await register(client)
    return {
        "user": data["user"],
        "tokens": data["tokens"],
        "password": data["password"],
        "headers": bearer(data["tokens"]),
    }


@pytest.fixture
async def admin_account(client: httpx.AsyncClient, db) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    data = await register(client, email=unique_email("admin"))
    from app.modules.auth.repository import UserRepository

    await UserRepository(db).update(data["user"]["id"], {"role": "admin"})
    # Re-login so the access token carries the admin role claim.
    response = await client.post(
        f"{API}/auth/login", json={"email": data["user"]["email"], "password": data["password"]}
    )
    assert response.status_code == 200, response.text
    tokens = response.json()["tokens"]
    return {"user": data["user"], "tokens": tokens, "password": data["password"], "headers": bearer(tokens)}
