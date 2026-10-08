"""Module 8 - charts, research assistant (heuristic + LLM tool loop), insights."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.db.collections import Collections
from app.integrations.llm import ChatResult, ToolCall
from app.modules.market.seed import seed_synthetic_market
from app.modules.research.assistant import ResearchAssistant
from app.modules.research.insights import compute_insight, period_bounds
from app.modules.research.models import InsightPeriod
from app.modules.research.service import ResearchService
from tests.conftest import API


@pytest.fixture(scope="module")
async def research_seed(db):  # type: ignore[no-untyped-def]
    await seed_synthetic_market(db, ["BTC", "ETH"], intervals=("1h", "1d"), counts={"1h": 300, "1d": 90})
    now = datetime.now(UTC)
    await db[Collections.SENTIMENT_SNAPSHOTS].insert_many(
        [
            {
                "ts": now - timedelta(hours=h),
                "meta": {"scope": "asset", "symbol": "BTC"},
                "score": 0.4,
                "index": 70.0 - h,
                "label": "bullish",
                "sample_size": 12,
            }
            for h in range(0, 30, 6)
        ]
        + [
            {
                "ts": now - timedelta(minutes=5),
                "meta": {"scope": "global", "symbol": None},
                "score": 0.3,
                "index": 65.0,
                "label": "bullish",
                "sample_size": 40,
            }
        ]
    )
    await db[Collections.WHALE_TRANSFERS].insert_many(
        [
            {
                "tx_hash": f"0x{'ab' * 31}{i:02d}",
                "log_index": -1,
                "symbol": "BTC",
                "amount": 100 + i,
                "amount_usd": 6_400_000 + i,
                "flow": "exchange_inflow" if i % 2 else "exchange_outflow",
                "risk_level": "low",
                "from_label": None,
                "to_label": "Binance",
                "block_time": now - timedelta(hours=i),
                "from_address": "0x" + "1" * 40,
                "to_address": "0x" + "2" * 40,
                "commentary": "test",
            }
            for i in range(6)
        ]
    )
    await db[Collections.PREDICTIONS].insert_one(
        {
            "symbol": "BTC",
            "horizon": "24h",
            "target_price": 70_000.0,
            "lower": 66_000.0,
            "upper": 74_000.0,
            "confidence": 0.81,
            "direction": "up",
            "change_pct": 4.2,
            "model": "statistical",
            "created_at": now,
            "target_time": now + timedelta(hours=24),
            "evaluated": False,
        }
    )
    return True


def test_period_bounds() -> None:
    start, end = period_bounds(InsightPeriod.MONTHLY, datetime(2026, 2, 14, tzinfo=UTC))
    assert (start, end) == (datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC))
    q_start, q_end = period_bounds(InsightPeriod.QUARTERLY, datetime(2026, 11, 30, tzinfo=UTC))
    assert (q_start, q_end) == (datetime(2026, 10, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC))


async def test_chart_series(client, research_seed) -> None:  # type: ignore[no-untyped-def]
    response = await client.get(
        f"{API}/research/charts/btc",
        params={"metrics": "price,volume,sentiment,netflow,predictions,whale_volume", "interval": "1h", "range": "7d"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "BTC"
    assert len(body["series"]["price"]) > 24
    assert len(body["series"]["sentiment"]) >= 1
    assert body["series"]["netflow"] and body["series"]["whale_volume"]
    assert body["series"]["predictions"][0]["target"] == 70_000.0
    assert "from" in body and "to" in body
    assert (await client.get(f"{API}/research/charts/BTC", params={"metrics": "price,bogus"})).status_code == 422


async def test_heuristic_assistant_plans_and_renders(db, research_seed) -> None:  # type: ignore[no-untyped-def]
    from app.core.config import get_settings
    from app.modules.market.models import Asset
    from app.modules.market.symbols import SymbolDetector

    detector = SymbolDetector([Asset(symbol="BTC", name="Bitcoin"), Asset(symbol="ETH", name="Ethereum")])
    assistant = ResearchAssistant(db, get_settings(), llm=None, detector=detector)
    assert assistant.mode == "heuristic"

    plan = dict(assistant.plan_tools("What's the forecast and whale activity for Bitcoin?", {}))
    assert {"get_price", "get_candles_summary", "get_prediction", "get_whale_activity"} <= set(plan)
    assert plan["get_prediction"] == {"symbol": "BTC"}

    reply = await assistant.respond([], "How is BTC doing and what do whales think?")
    assert reply.model == "heuristic"
    assert "BTC" in reply.content and "Whales" in reply.content
    assert reply.data["get_price"]["price"] is not None
    assert reply.data["get_whale_activity"]["transfers"] == 6

    wallet_reply = await assistant.respond([], "Is 0x1111111111111111111111111111111111111111 a safe wallet?")
    assert wallet_reply.data["check_wallet"]["valid"] is True
    assert "not on the blacklist" in wallet_reply.content


async def test_llm_assistant_tool_loop(db, research_seed) -> None:  # type: ignore[no-untyped-def]
    from app.core.config import get_settings

    class FakeLLM:
        available = True
        total_tokens = 0

        def __init__(self) -> None:
            self.calls: list[list[dict[str, Any]]] = []

        async def chat(self, messages, *, tools=None, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(messages)
            if len(self.calls) == 1:
                assert tools and any(t["function"]["name"] == "get_price" for t in tools)
                return ChatResult(
                    content=None,
                    tool_calls=[ToolCall(id="call_1", name="get_price", arguments={"symbol": "BTC"})],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                )
            tool_message = messages[-1]
            assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_1"
            payload = json.loads(tool_message["content"])
            return ChatResult(
                content=f"BTC trades at {payload['price']:.0f} according to the price tool.",
                usage={"prompt_tokens": 20, "completion_tokens": 8},
            )

    fake = FakeLLM()
    assistant = ResearchAssistant(db, get_settings(), llm=fake)  # type: ignore[arg-type]
    assert assistant.mode == "gpt-4o"
    reply = await assistant.respond(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], "Price of BTC?"
    )
    assert reply.model == "gpt-4o"
    assert "according to the price tool" in reply.content
    assert reply.tool_calls == [{"tool": "get_price", "arguments": {"symbol": "BTC"}}]
    assert reply.tokens == 43
    assert fake.calls[0][0]["role"] == "system"


async def test_chat_sessions_api_and_ownership(client, user_account, research_seed) -> None:  # type: ignore[no-untyped-def]
    headers = user_account["headers"]
    created = await client.post(
        f"{API}/research/chat/sessions", json={"context": {"symbols": ["ETH"]}}, headers=headers
    )
    assert created.status_code == 201
    session_id = created.json()["id"]

    reply = await client.post(
        f"{API}/research/chat/sessions/{session_id}/messages",
        json={"content": "Give me a quick summary of the market and sentiment"},
        headers=headers,
    )
    assert reply.status_code == 200
    body = reply.json()
    assert body["assistant_message"]["role"] == "assistant"
    assert body["assistant_message"]["content"]
    assert body["session"]["message_count"] == 2
    assert body["session"]["title"].startswith("Give me a quick summary")

    detail = await client.get(f"{API}/research/chat/sessions/{session_id}", headers=headers)
    assert detail.status_code == 200
    assert [m["role"] for m in detail.json()["messages"]] == ["user", "assistant"]

    pinned = await client.patch(f"{API}/research/chat/sessions/{session_id}", json={"pinned": True}, headers=headers)
    assert pinned.json()["pinned"] is True
    listing = await client.get(f"{API}/research/chat/sessions", headers=headers)
    assert listing.json()[0]["id"] == session_id

    other = await client.post(
        f"{API}/auth/register", json={"email": f"other-{session_id[:6]}@example.com", "password": "Str0ngPass!"}
    )
    other_headers = {"Authorization": f"Bearer {other.json()['tokens']['access_token']}"}
    assert (await client.get(f"{API}/research/chat/sessions/{session_id}", headers=other_headers)).status_code == 404

    quick = await client.post(f"{API}/research/chat/quick", json={"content": "Any fraud alerts?"}, headers=headers)
    assert quick.status_code == 200
    assert "get_fraud_alerts" in quick.json()["assistant_message"]["data"]

    library = await client.get(f"{API}/research/library", headers=headers)
    assert library.status_code == 200
    assert library.json()["assistant_mode"] == "heuristic"
    assert len(library.json()["sessions"]) >= 2
    pulse = await client.get(f"{API}/research/pulse", headers=headers)
    assert pulse.status_code == 200 and pulse.json()["messages_last_hour"] >= 2

    assert (await client.delete(f"{API}/research/chat/sessions/{session_id}", headers=headers)).status_code == 204
    assert (await client.get(f"{API}/research/chat/sessions/{session_id}", headers=headers)).status_code == 404


async def test_insights_compute_and_endpoints(client, db, admin_account, user_account, research_seed) -> None:  # type: ignore[no-untyped-def]
    from app.core.config import get_settings

    insight = await compute_insight(db, get_settings(), InsightPeriod.MONTHLY, llm=None)
    assert insight.in_progress is True
    assert insight.metrics["assets_covered"] >= 1
    assert insight.metrics["whale_transfers"] >= 6
    assert insight.summary and insight.generated_by == "template"

    denied = await client.post(f"{API}/research/insights/refresh", headers=user_account["headers"])
    assert denied.status_code == 403
    refreshed = await client.post(f"{API}/research/insights/refresh", headers=admin_account["headers"])
    assert refreshed.status_code == 200
    periods = {i["period"] for i in refreshed.json()}
    assert periods == {"monthly", "quarterly"}

    listing = await client.get(f"{API}/research/insights", params={"period": "monthly", "limit": 3})
    assert listing.status_code == 200
    assert listing.json()[0]["title"].startswith("Monthly Market Insight")

    service = ResearchService(db, get_settings(), llm=None)
    pulse = await service.pulse()
    assert pulse.assistant_mode == "heuristic"
