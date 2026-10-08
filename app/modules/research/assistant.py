"""Multi-turn AI research assistant (FE-3).

With an OpenAI key the assistant runs a GPT-4o tool-calling loop over the data
tools in ``tools.py``. Without one, a rule-based router picks the tools from the
question and renders a templated analyst answer, so the feature always works.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.integrations.evm import is_address
from app.integrations.llm import LLMClient
from app.modules.market.symbols import SymbolDetector
from app.modules.research.tools import TOOLS, run_tool

logger = structlog.get_logger(__name__)

MAX_TOOL_ROUNDS = 5
HISTORY_LIMIT = 20

SYSTEM_PROMPT = """You are QuantaAI Research Assistant, the analyst inside the Quantachain crypto-intelligence platform.
Answer questions about crypto markets using the provided tools for live data (prices, candles, sentiment, news,
whale flows, forecasts, volatility, fraud alerts, wallet checks, contract scans). Always call tools before stating
numbers and cite which data you used. Be concise, structured (short paragraphs or bullets), quantitative, and
explicit about uncertainty. Mention the time the data refers to. Never promise returns; end market-direction answers
with a one-line risk note. If a tool reports data is unavailable, say so plainly."""


@dataclass
class AssistantReply:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    model: str = "heuristic"
    tokens: int | None = None
    latency_ms: float | None = None


class ResearchAssistant:
    def __init__(
        self,
        db: AsyncDatabase[dict[str, Any]],
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        detector: SymbolDetector | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.llm = llm
        self.detector = detector

    @property
    def mode(self) -> str:
        return "gpt-4o" if self.llm is not None and self.llm.available else "heuristic"

    async def respond(
        self, history: list[dict[str, str]], question: str, *, context: dict[str, Any] | None = None
    ) -> AssistantReply:
        started = time.perf_counter()
        if self.llm is not None and self.llm.available:
            try:
                reply = await self._respond_llm(history, question, context or {})
            except ExternalServiceError as exc:
                logger.warning("assistant_llm_failed_falling_back", error=str(exc))
                reply = await self._respond_heuristic(question, context or {})
                reply.content = "(AI model unavailable, showing data-driven summary)\n\n" + reply.content
        else:
            reply = await self._respond_heuristic(question, context or {})
        reply.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return reply

    # ----------------------------------------------------------- LLM path
    async def _respond_llm(
        self, history: list[dict[str, str]], question: str, context: dict[str, Any]
    ) -> AssistantReply:
        assert self.llm is not None
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT + self._context_hint(context)}]
        messages.extend(history[-HISTORY_LIMIT:])
        messages.append({"role": "user", "content": question})
        tool_defs = [spec.openai_schema() for spec in TOOLS.values()]
        calls_made: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        tokens = 0
        for _round in range(MAX_TOOL_ROUNDS):
            result = await self.llm.chat(messages, tools=tool_defs)
            tokens += sum(result.usage.values()) if result.usage else 0
            if not result.tool_calls:
                return AssistantReply(
                    content=(result.content or "").strip() or "I could not produce an answer.",
                    tool_calls=calls_made,
                    data=data,
                    model=self.mode,
                    tokens=tokens,
                )
            messages.append(result.as_assistant_message())
            for call in result.tool_calls:
                output = await run_tool(self.db, self.settings, call.name, call.arguments)
                calls_made.append({"tool": call.name, "arguments": call.arguments})
                data[call.name] = output
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output, default=str)[:12000]}
                )
        final = await self.llm.chat(messages, tools=None)
        tokens += sum(final.usage.values()) if final.usage else 0
        return AssistantReply(
            content=(final.content or "").strip() or "Here is what the data shows.",
            tool_calls=calls_made,
            data=data,
            model=self.mode,
            tokens=tokens,
        )

    @staticmethod
    def _context_hint(context: dict[str, Any]) -> str:
        symbols = context.get("symbols")
        return f"\nThe user is currently focused on: {', '.join(symbols)}." if symbols else ""

    # ----------------------------------------------------- heuristic path
    async def _respond_heuristic(self, question: str, context: dict[str, Any]) -> AssistantReply:
        plan = self.plan_tools(question, context)
        data: dict[str, Any] = {}
        calls: list[dict[str, Any]] = []
        for name, args in plan:
            data[name] = await run_tool(self.db, self.settings, name, args)
            calls.append({"tool": name, "arguments": args})
        return AssistantReply(content=self.render(question, data), tool_calls=calls, data=data, model="heuristic")

    def plan_tools(self, question: str, context: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        q = question.lower()
        symbols = self.detector.detect(question) if self.detector else []
        if not symbols and context.get("symbols"):
            symbols = list(context["symbols"])
        symbol = symbols[0] if symbols else "BTC"
        plan: list[tuple[str, dict[str, Any]]] = []
        address_match = re.search(r"0x[a-fA-F0-9]{40}", question)

        if address_match and is_address(address_match.group(0)):
            address = address_match.group(0)
            if any(k in q for k in ("contract", "token", "scan", "rug", "honeypot", "safe", "audit")):
                plan.append(("scan_contract", {"address": address}))
            plan.append(("check_wallet", {"address": address}))
            return plan
        if any(k in q for k in ("scam", "rug", "fraud", "pump", "dump", "exploit", "hack", "alert")):
            plan.append(("get_fraud_alerts", {"limit": 8}))
        if any(
            k in q for k in ("whale", "on-chain", "onchain", "netflow", "inflow", "outflow", "smart money", "wallet")
        ):
            plan.append(("get_whale_activity", {"symbol": symbol, "hours": 24}))
        if any(k in q for k in ("predict", "forecast", "target", "will", "next", "outlook", "price in")):
            plan.append(("get_prediction", {"symbol": symbol}))
            plan.append(("get_volatility", {"symbol": symbol}))
        if any(k in q for k in ("sentiment", "vibe", "mood", "fear", "greed", "bullish", "bearish", "feel")):
            plan.append(("get_sentiment", {"symbol": symbol} if symbols else {}))
        if any(k in q for k in ("news", "headline", "happen", "geopolit", "regulat", "sec ", "etf", "why")):
            plan.append(("get_news", {"symbol": symbol, "limit": 6} if symbols else {"limit": 6}))
        if any(k in q for k in ("trend", "narrative", "topic", "hot")):
            plan.append(("get_trending_topics", {"hours": 24, "limit": 8}))
        if any(k in q for k in ("volatil", "risk", "regime")) and ("get_volatility", {"symbol": symbol}) not in plan:
            plan.append(("get_volatility", {"symbol": symbol}))
        if symbols or any(
            k in q
            for k in (
                "price",
                "chart",
                "rsi",
                "trading at",
                "how is",
                "performance",
                "summary",
                "report",
                "synthesis",
                "overview",
            )
        ):
            if symbols:
                plan.insert(0, ("get_price", {"symbol": symbol}))
                plan.insert(1, ("get_candles_summary", {"symbol": symbol, "interval": "1h", "lookback": 168}))
            else:
                plan.insert(0, ("get_market_overview", {"limit": 5}))
        if not plan:
            plan = [("get_market_overview", {"limit": 5}), ("get_sentiment", {}), ("get_news", {"limit": 5})]
        # Deduplicate while keeping order.
        seen: set[str] = set()
        unique: list[tuple[str, dict[str, Any]]] = []
        for name, args in plan:
            if name not in seen:
                unique.append((name, args))
                seen.add(name)
        return unique

    def render(self, question: str, data: dict[str, Any]) -> str:
        lines: list[str] = []
        price = data.get("get_price")
        summary = data.get("get_candles_summary")
        if price and price.get("price") is not None:
            change = price.get("change_24h_pct")
            change_text = f" ({change:+.2f}% 24h)" if isinstance(change, int | float) else ""
            lines.append(
                f"**{price['symbol']}** is trading at ${price['price']:,.2f}{change_text}"
                + (f" (source: {price.get('source')})." if price.get("source") else ".")
            )
        if summary and summary.get("available"):
            lines.append(
                f"Over the last {summary['candles']} x {summary['interval']} candles it moved {summary['change_pct']:+.2f}% "
                f"(range ${summary['low']:,.2f} - ${summary['high']:,.2f}); RSI(14) is {summary['rsi_14']} and per-bar volatility {summary['volatility_per_bar_pct']}%."
            )
        overview = data.get("get_market_overview")
        if overview:
            pulse = overview.get("pulse", {})
            if pulse.get("fear_greed_value") is not None:
                lines.append(
                    f"Market pulse: Fear & Greed {pulse['fear_greed_value']} ({pulse.get('fear_greed_label')}), BTC dominance {pulse.get('btc_dominance_pct')}%."
                )
            rows = overview.get("assets", [])[:5]
            if rows:
                lines.append(
                    "Top assets: "
                    + ", ".join(f"{r['symbol']} ${r['price']:,.2f}" if r.get("price") else r["symbol"] for r in rows)
                    + "."
                )
        sentiment = data.get("get_sentiment")
        if sentiment:
            if sentiment.get("available"):
                scope = sentiment.get("symbol") or "the market"
                lines.append(
                    f"Sentiment for {scope}: index {sentiment['index']} ({str(sentiment.get('label', '')).replace('_', ' ')}) from {sentiment.get('sample_size')} items."
                )
            else:
                lines.append("Sentiment: no snapshot available yet.")
        whales = data.get("get_whale_activity")
        if whales:
            if whales.get("available"):
                net = whales["exchange_netflow_usd"]
                bias = (
                    "net exchange inflow (potential sell pressure)"
                    if net > 0
                    else "net exchange outflow (accumulation)"
                    if net < 0
                    else "balanced flows"
                )
                lines.append(
                    f"Whales: {whales['transfers']} large transfers worth ${whales['volume_usd']:,.0f} in {whales['hours']}h; {bias} of ${abs(net):,.0f}."
                )
            else:
                lines.append("Whales: no large transfers recorded in the window.")
        prediction = data.get("get_prediction")
        if prediction:
            if prediction.get("available"):
                parts = [
                    f"{h['horizon']}: ${h['target_price']:,.2f} ({h['change_pct']:+.2f}%, conf {round((h.get('confidence') or 0) * 100)}%)"
                    for h in prediction["horizons"]
                    if h.get("target_price") is not None
                ]
                lines.append("AI forecast - " + "; ".join(parts) + ".")
            else:
                lines.append("AI forecast: not available yet for this asset.")
        vol = data.get("get_volatility")
        if vol and vol.get("available"):
            regime = vol.get("regime") or vol.get("regime_label")
            lines.append(f"Volatility regime: {regime}." if regime else "Volatility forecast available.")
        news = data.get("get_news")
        if news and news.get("articles"):
            lines.append("Recent headlines:")
            for article in news["articles"][:5]:
                tag = f" [{article['sentiment']}]" if article.get("sentiment") else ""
                lines.append(f"- {article['title']} ({article.get('source')}){tag}")
        trending = data.get("get_trending_topics")
        if trending and trending.get("topics"):
            lines.append(
                "Trending topics: " + ", ".join(f"#{t['topic']} ({t['count']})" for t in trending["topics"][:6]) + "."
            )
        alerts = data.get("get_fraud_alerts")
        if alerts is not None:
            if alerts.get("alerts"):
                lines.append("Latest risk alerts:")
                for alert in alerts["alerts"][:5]:
                    lines.append(f"- [{alert.get('severity')}] {alert.get('title')}")
            else:
                lines.append("No fraud alerts recorded recently.")
        wallet = data.get("check_wallet")
        if wallet:
            if not wallet.get("valid"):
                lines.append("That does not look like a valid EVM address.")
            else:
                status = "is BLACKLISTED" if wallet.get("blacklisted") else "is not on the blacklist"
                lines.append(
                    f"Wallet {wallet['address'][:10]}... {status}; seen in {wallet.get('whale_transfers_seen', 0)} whale transfers."
                )
        scan = data.get("scan_contract")
        if scan:
            if scan.get("available") and scan.get("risk_score") is not None:
                lines.append(
                    f"Contract scan: risk score {scan['risk_score']}/100 ({scan.get('risk_label')}), {len(scan.get('findings', []))} findings."
                )
            elif scan.get("note"):
                lines.append(f"Contract scan: {scan['note']}")
        if not lines:
            lines.append(
                "I could not find data for that question yet. Try asking about a tracked asset such as BTC or ETH."
            )
        lines.append("")
        lines.append("_Data-driven summary (heuristic mode). Not financial advice._")
        return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT", "AssistantReply", "ResearchAssistant"]
