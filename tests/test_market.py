"""Module 2 - market data endpoints, candle roll-ups, dedup and symbol detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.integrations.news import canonical_url, content_hash, sanitize_text, url_hash
from app.modules.market.models import Asset, Candle, NewsArticle, SocialPost
from app.modules.market.repository import CandleRepository, NewsRepository, SocialRepository
from app.modules.market.seed import generate_synthetic_candles, seed_synthetic_market
from app.modules.market.symbols import SymbolDetector
from tests.conftest import API


@pytest.fixture(scope="module")
async def seeded(db):  # type: ignore[no-untyped-def]
    written = await seed_synthetic_market(
        db, ["BTC", "ETH"], intervals=("1m", "1h", "1d"), counts={"1m": 120, "1h": 200, "1d": 60}
    )
    assert written["1h"] >= 200
    return written


# --------------------------------------------------------------- unit level
def test_sanitize_and_canonical_url() -> None:
    assert sanitize_text("<p>Hello &amp; <b>world</b></p>\n\n  x") == "Hello & world x"
    assert sanitize_text(None) is None
    url = "https://Example.com/news/item/?utm_source=x&id=5&fbclid=abc"
    assert canonical_url(url) == "https://example.com/news/item?id=5"
    assert url_hash(url) == url_hash("https://example.com/news/item?id=5")
    assert content_hash("Bitcoin hits NEW high!!") == content_hash("bitcoin hits new high")


def test_symbol_detector() -> None:
    assets = [
        Asset(symbol="BTC", name="Bitcoin"),
        Asset(symbol="ETH", name="Ethereum"),
        Asset(symbol="NEAR", name="NEAR Protocol"),
        Asset(symbol="SOL", name="Solana"),
    ]
    detector = SymbolDetector(assets)
    assert detector.detect("Bitcoin rallies as $ETH lags; Solana ETF chatter grows") == ["ETH", "BTC", "SOL"]
    # Lower-case 'near' is an English word and must not match; the uppercase ticker does.
    assert detector.detect("prices are near the top") == []
    assert detector.detect("NEAR protocol upgrade") == ["NEAR"]
    assert detector.detect("ethbtc pair") == []


def test_synthetic_candles_are_consistent() -> None:
    candles = generate_synthetic_candles("BTC", "1h", 50, seed=1)
    assert len(candles) == 50
    for c in candles:
        assert c.low <= min(c.open, c.close) <= max(c.open, c.close) <= c.high
    assert candles[-1].closed is False and candles[-2].closed is True
    assert candles[1].open_time - candles[0].open_time == timedelta(hours=1)


# ----------------------------------------------------------- candle rollup
async def test_minute_candles_roll_up_into_hour_bucket(db) -> None:  # type: ignore[no-untyped-def]
    repo = CandleRepository(db)
    hour = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
    prices = [100.0, 101.0, 99.0, 102.0, 101.5]
    for i, price in enumerate(prices):
        open_time = hour + timedelta(minutes=i)
        candle = Candle(
            symbol="TEST",
            interval="1m",
            open_time=open_time,
            close_time=open_time + timedelta(seconds=59, milliseconds=999),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price + 0.5,
            volume=10,
            quote_volume=1000,
            trades=5,
            closed=True,
        )
        _, newly_closed = await repo.upsert_candle(candle)
        assert newly_closed is True
        await repo.rollup(candle)
        # Replaying the same closed candle must not double count.
        _, again = await repo.upsert_candle(candle)
        assert again is False

    bucket = await repo.find_one({"symbol": "TEST", "interval": "1h", "open_time": hour})
    assert bucket is not None
    assert bucket.open == 100.0
    assert bucket.high == 103.0
    assert bucket.low == 98.0
    assert bucket.close == 102.0
    assert bucket.volume == 50
    assert bucket.trades == 25
    assert bucket.closed is False

    day = await repo.find_one({"symbol": "TEST", "interval": "1d", "open_time": datetime(2026, 3, 1, tzinfo=UTC)})
    assert day is not None and day.volume == 50


async def test_rebuild_open_buckets_from_minutes(db) -> None:  # type: ignore[no-untyped-def]
    repo = CandleRepository(db)
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    hour_start = now.replace(minute=0)
    minutes = [hour_start + timedelta(minutes=i) for i in range(0, min(now.minute, 5) + 1)]
    for i, open_time in enumerate(minutes):
        await repo.upsert_candle(
            Candle(
                symbol="REBUILD",
                interval="1m",
                open_time=open_time,
                close_time=open_time + timedelta(seconds=59),
                open=10 + i,
                high=11 + i,
                low=9 + i,
                close=10.5 + i,
                volume=1,
                quote_volume=10,
                trades=1,
            )
        )
    rebuilt = await repo.rebuild_open_buckets("REBUILD")
    assert rebuilt >= 1
    bucket = await repo.find_one({"symbol": "REBUILD", "interval": "1h", "open_time": hour_start})
    assert bucket is not None
    assert bucket.open == 10
    assert bucket.volume == len(minutes)
    assert bucket.close == 10.5 + len(minutes) - 1


# ----------------------------------------------------------------- HTTP API
async def test_overview_and_assets(client, seeded) -> None:  # type: ignore[no-untyped-def]
    response = await client.get(f"{API}/market/overview")
    assert response.status_code == 200
    body = response.json()
    assert "pulse" in body and "assets" in body
    symbols = {row["symbol"] for row in body["assets"]}
    assert {"BTC", "ETH"} <= symbols
    btc = next(row for row in body["assets"] if row["symbol"] == "BTC")
    assert btc["price"] is not None
    assert btc["sentiment"]["index"] is None  # no Module 4 snapshot yet

    for tab in ("hot", "gainers", "losers", "volume", "new", "market_cap"):
        tab_response = await client.get(f"{API}/market/assets", params={"tab": tab, "limit": 5})
        assert tab_response.status_code == 200, tab
        assert len(tab_response.json()) >= 2
    assert (await client.get(f"{API}/market/assets", params={"tab": "bogus"})).status_code == 422


async def test_asset_detail_ticker_and_candles(client, seeded) -> None:  # type: ignore[no-untyped-def]
    detail = await client.get(f"{API}/market/assets/btc")
    assert detail.status_code == 200
    assert detail.json()["asset"]["symbol"] == "BTC"
    assert detail.json()["latest_candle"]["c"] > 0

    ticker = await client.get(f"{API}/market/assets/BTC/ticker")
    assert ticker.status_code == 200
    assert ticker.json()["price"] > 0

    candles = await client.get(f"{API}/market/assets/BTC/candles", params={"interval": "1h", "limit": 50})
    assert candles.status_code == 200
    payload = candles.json()
    assert payload["count"] == 50
    times = [c["t"] for c in payload["candles"]]
    assert times == sorted(times)

    end = payload["candles"][10]["t"]
    bounded = await client.get(f"{API}/market/assets/BTC/candles", params={"interval": "1h", "end": end, "limit": 5})
    assert bounded.status_code == 200
    assert bounded.json()["candles"][-1]["t"] == end

    assert (await client.get(f"{API}/market/assets/BTC/candles", params={"interval": "7m"})).status_code == 422
    assert (await client.get(f"{API}/market/assets/NOPE")).status_code == 404


async def test_news_and_social_endpoints(client, db) -> None:  # type: ignore[no-untyped-def]
    news_repo = NewsRepository(db)
    article = NewsArticle(
        url="https://example.com/sec-approves?utm_source=rss",
        url_hash=url_hash("https://example.com/sec-approves?utm_source=rss"),
        title="SEC approves spot Bitcoin ETF options",
        summary="Regulator clears the way for options on BTC ETFs.",
        source_name="Example Wire",
        published_at=datetime.now(UTC),
        fetched_at=datetime.now(UTC),
        origin="rss",
        symbols=["BTC"],
    )
    assert await news_repo.insert_if_new(article) is True
    assert await news_repo.insert_if_new(article.model_copy(update={"id": None})) is False  # same canonical URL

    response = await client.get(f"{API}/market/news", params={"symbol": "btc", "limit": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    item = next(i for i in body["items"] if i["title"].startswith("SEC approves"))
    assert item["analysis"]["status"] == "pending"
    single = await client.get(f"{API}/market/news/{item['id']}")
    assert single.status_code == 200
    search = await client.get(f"{API}/market/news", params={"q": "ETF"})
    assert search.status_code == 200 and search.json()["total"] >= 1

    social_repo = SocialRepository(db)
    post = SocialPost(
        platform="reddit",
        external_id="reddit:abc123",
        content_hash=content_hash("ETH is going to flip BTC"),
        title="ETH is going to flip BTC",
        text="",
        url="https://reddit.com/r/x/abc123",
        posted_at=datetime.now(UTC),
        community="CryptoCurrency",
        symbols=["ETH", "BTC"],
    )
    assert await social_repo.insert_if_new(post) is True
    duplicate = post.model_copy(
        update={"id": None, "external_id": "reddit:other", "title": "eth is going to flip btc!!"}
    )
    duplicate.content_hash = content_hash("eth is going to flip btc!!")
    assert await social_repo.insert_if_new(duplicate) is False
    social = await client.get(f"{API}/market/social", params={"platform": "reddit"})
    assert social.status_code == 200
    assert social.json()["total"] >= 1


async def test_ingestion_hub_requires_admin(client, user_account, admin_account) -> None:  # type: ignore[no-untyped-def]
    assert (await client.get(f"{API}/market/ingestion", headers=user_account["headers"])).status_code == 403
    response = await client.get(f"{API}/market/ingestion", headers=admin_account["headers"])
    assert response.status_code == 200
    assert response.json()["worker_online"] is False
    logs = await client.get(f"{API}/market/ingestion/logs", headers=admin_account["headers"])
    assert logs.status_code == 200
    missing = await client.post(
        f"{API}/market/ingestion/sources/nothing", json={"enabled": False}, headers=admin_account["headers"]
    )
    assert missing.status_code == 404


async def test_admin_can_track_and_untrack_assets(client, admin_account, user_account) -> None:  # type: ignore[no-untyped-def]
    forbidden = await client.post(f"{API}/market/assets", json={"symbol": "ARB"}, headers=user_account["headers"])
    assert forbidden.status_code == 403
    created = await client.post(f"{API}/market/assets", json={"symbol": "arb"}, headers=admin_account["headers"])
    assert created.status_code == 201
    assert created.json()["symbol"] == "ARB"
    assert created.json()["coingecko_id"] == "arbitrum"
    conflict = await client.post(f"{API}/market/assets", json={"symbol": "ARB"}, headers=admin_account["headers"])
    assert conflict.status_code == 409
    assert (await client.delete(f"{API}/market/assets/ARB", headers=admin_account["headers"])).status_code == 204
    rows = await client.get(f"{API}/market/assets", params={"tab": "new", "limit": 100})
    assert "ARB" not in {r["symbol"] for r in rows.json()}
