"""Unit tests for framework-level helpers (no database needed)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pyotp
import pytest

from app.core.events import EventBus, Topics
from app.core.rate_limit import SlidingWindowLimiter
from app.core.security import (
    SecretBox,
    TokenError,
    create_token,
    decode_token,
    generate_api_key,
    hash_password,
    verify_password,
    verify_totp,
)
from app.core.timeutils import floor_time, interval_seconds


def test_password_hashing_roundtrip() -> None:
    hashed = hash_password("Str0ngPass!")
    assert hashed != "Str0ngPass!"
    assert verify_password("Str0ngPass!", hashed)
    assert not verify_password("wrong", hashed)
    assert not verify_password("anything", None)


def test_jwt_roundtrip_and_type_check(test_env) -> None:  # type: ignore[no-untyped-def]
    token = create_token("user-1", "access", expires_in=timedelta(minutes=5), claims={"role": "admin"})
    payload = decode_token(token, "access")
    assert payload["sub"] == "user-1"
    assert payload["role"] == "admin"
    with pytest.raises(TokenError):
        decode_token(token, "refresh")


def test_expired_jwt_rejected(test_env) -> None:  # type: ignore[no-untyped-def]
    token = create_token("user-1", "access", expires_in=timedelta(seconds=-5))
    with pytest.raises(TokenError):
        decode_token(token)


def test_totp_verification() -> None:
    secret = pyotp.random_base32()
    assert verify_totp(secret, pyotp.TOTP(secret).now())
    assert not verify_totp(secret, "000000") or pyotp.TOTP(secret).now() == "000000"
    assert not verify_totp(secret, "abc")


def test_secret_box_roundtrip(test_env) -> None:  # type: ignore[no-untyped-def]
    box = SecretBox()
    ciphertext = box.encrypt("binance-secret")
    assert ciphertext != "binance-secret"
    assert box.decrypt(ciphertext) == "binance-secret"


def test_api_key_generation() -> None:
    key, prefix = generate_api_key()
    assert key.startswith("qc_")
    assert key.startswith(prefix)
    assert len(key) > 30


def test_interval_helpers() -> None:
    assert interval_seconds("1m") == 60
    assert interval_seconds("4h") == 14400
    assert interval_seconds("1d") == 86400
    ts = datetime(2026, 1, 1, 13, 37, 42, tzinfo=UTC)
    assert floor_time(ts, "1h") == datetime(2026, 1, 1, 13, 0, tzinfo=UTC)
    assert floor_time(ts, "15m") == datetime(2026, 1, 1, 13, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        interval_seconds("fortnight")


def test_sliding_window_limiter() -> None:
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60)
    results = [limiter.check("k")[0] for _ in range(4)]
    assert results == [True, True, True, False]
    assert limiter.check("other")[0] is True


async def test_event_bus_fanout_and_drop_oldest() -> None:
    bus = EventBus()
    sub = bus.subscribe(Topics.PRICE_TICK, maxsize=2)
    other = bus.subscribe(Topics.WHALE_TRANSFER)
    for i in range(3):
        bus.publish(Topics.PRICE_TICK, {"i": i})
    bus.publish(Topics.SIGNAL, {"ignored": True})
    assert sub.dropped == 1
    first = await sub.get(timeout=0.1)
    second = await sub.get(timeout=0.1)
    assert first is not None and second is not None
    assert [first.payload["i"], second.payload["i"]] == [1, 2]
    assert await other.get(timeout=0.01) is None
    sub.close()
    assert bus.subscriber_count == 1
    await asyncio.sleep(0)
