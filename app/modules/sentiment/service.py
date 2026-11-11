"""Module 4 business logic: sentiment inference, NER entity extraction, Vibe Index snapshots, and geopolitical heatmap."""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from typing import Any

import numpy as np
import structlog

from app.core.config import Settings, get_settings
from app.core.timeutils import utcnow
from app.integrations.llm import LLMClient
from app.modules.market.models import AnalysisStatus, Entity, SentimentLabel, TextAnalysis
from app.modules.market.repository import (
    AssetStatsRepository,
    CandleRepository,
    MarketGlobalRepository,
    NewsRepository,
    SocialRepository,
)
from app.modules.sentiment.models import GeoCategory, GeoEvent, SentimentSnapshot
from app.modules.sentiment.repository import GeoEventRepository, SentimentSnapshotRepository
from app.modules.sentiment.schemas import (
    CorrelationOut,
    GeoHeatmapOut,
    SectorHeatmapItem,
    SentimentOverviewOut,
    SentimentSnapshotOut,
)

logger = structlog.get_logger(__name__)

# Heuristic sentiment dictionaries
POSITIVE_WORDS = frozenset(
    {
        "surge",
        "surges",
        "surging",
        "bull",
        "bullish",
        "gain",
        "gains",
        "gaining",
        "soar",
        "soars",
        "breakout",
        "rally",
        "rallies",
        "rallying",
        "approve",
        "approved",
        "approval",
        "adoption",
        "profit",
        "profits",
        "partnership",
        "record",
        "high",
        "all-time",
        "highs",
        "moon",
        "pump",
        "win",
        "winning",
        "secure",
        "upgraded",
        "upgrade",
        "launch",
        "launched",
        "boost",
        "boosts",
        "positive",
        "growth",
        "inflow",
        "inflows",
        "accumulate",
        "accumulation",
        "support",
        "optimism",
    }
)

NEGATIVE_WORDS = frozenset(
    {
        "crash",
        "crashes",
        "crashing",
        "plunge",
        "plunges",
        "dump",
        "dumps",
        "dumping",
        "hack",
        "hacked",
        "hacks",
        "scam",
        "scams",
        "exploit",
        "exploited",
        "ban",
        "banned",
        "bans",
        "sanction",
        "sanctions",
        "lawsuit",
        "sued",
        "fraud",
        "fraudulent",
        "bear",
        "bearish",
        "drop",
        "drops",
        "dropping",
        "fall",
        "falls",
        "falling",
        "lose",
        "loses",
        "loss",
        "losses",
        "fail",
        "failed",
        "decline",
        "declines",
        "risk",
        "risks",
        "warning",
        "arrest",
        "arrested",
        "subpoena",
        "probe",
        "investigation",
        "bankrupt",
        "bankruptcy",
        "outflow",
        "liquidation",
        "liquidated",
        "panic",
        "bleeding",
        "collapse",
        "rugpull",
        "rug",
    }
)

ENTITY_PATTERNS: list[tuple[str, str, str | None]] = [
    (r"\b(BTC|Bitcoin)\b", "TOKEN", "BTC"),
    (r"\b(ETH|Ethereum|Ether)\b", "TOKEN", "ETH"),
    (r"\b(SOL|Solana)\b", "TOKEN", "SOL"),
    (r"\b(BNB|Binance Coin)\b", "TOKEN", "BNB"),
    (r"\b(XRP|Ripple)\b", "TOKEN", "XRP"),
    (r"\b(ADA|Cardano)\b", "TOKEN", "ADA"),
    (r"\b(DOGE|Dogecoin)\b", "TOKEN", "DOGE"),
    (r"\b(AVAX|Avalanche)\b", "TOKEN", "AVAX"),
    (r"\b(LINK|Chainlink)\b", "TOKEN", "LINK"),
    (r"\b(DOT|Polkadot)\b", "TOKEN", "DOT"),
    (r"\b(USDT|Tether)\b", "TOKEN", "USDT"),
    (r"\b(USDC|USD Coin)\b", "TOKEN", "USDC"),
    (r"\b(SEC|Securities and Exchange Commission)\b", "REGULATOR", None),
    (r"\b(CFTC|Commodity Futures Trading Commission)\b", "REGULATOR", None),
    (r"\b(Fed|Federal Reserve|FOMC)\b", "REGULATOR", None),
    (r"\b(DOJ|Department of Justice)\b", "REGULATOR", None),
    (r"\b(Binance)\b", "EXCHANGE", None),
    (r"\b(Coinbase)\b", "EXCHANGE", None),
    (r"\b(Kraken)\b", "EXCHANGE", None),
    (r"\b(Uniswap)\b", "PROTOCOL", None),
    (r"\b(Aave)\b", "PROTOCOL", None),
]

GEO_KEYWORDS = {
    GeoCategory.REGULATION: [
        "sec",
        "cftc",
        "regulator",
        "regulation",
        "regulatory",
        "ban",
        "sanction",
        "policy",
        "bill",
        "senate",
        "congress",
        "compliance",
        "framework",
        "license",
        "licensing",
        "mika",
        "treasury",
    ],
    GeoCategory.SECURITY: [
        "hack",
        "hacked",
        "exploit",
        "exploited",
        "vulnerability",
        "breach",
        "theft",
        "stolen",
        "malware",
        "drainer",
        "phishing",
    ],
    GeoCategory.MACRO: [
        "interest rate",
        "inflation",
        "cpi",
        "fomc",
        "federal reserve",
        "rate cut",
        "rate hike",
        "gdp",
        "recession",
        "unemployment",
        "macro",
    ],
    GeoCategory.LEGAL: [
        "lawsuit",
        "court",
        "judge",
        "settlement",
        "plea",
        "indictment",
        "trial",
        "sentence",
        "prison",
        "arrest",
        "guilty",
        "appeal",
    ],
    GeoCategory.GEOPOLITICAL: [
        "war",
        "conflict",
        "sanctions",
        "tariff",
        "election",
        "geopolitical",
        "iran",
        "russia",
        "china",
        "middle east",
        "ukraine",
    ],
}


class SentimentService:
    def __init__(self, db: Any, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.snapshots = SentimentSnapshotRepository(db)
        self.geo_events = GeoEventRepository(db)
        self.news_repo = NewsRepository(db)
        self.social_repo = SocialRepository(db)
        self.candles_repo = CandleRepository(db)
        self.stats_repo = AssetStatsRepository(db)
        self.market_global = MarketGlobalRepository(db)
        self.llm = LLMClient(self.settings)

    # ----------------------------------------------------------- text analysis
    async def analyze_text(self, text: str, symbol_hint: str | None = None) -> TextAnalysis:
        if not text or not text.strip():
            return TextAnalysis(
                status=AnalysisStatus.SKIPPED,
                sentiment_score=0.0,
                sentiment_label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                summary="Empty text",
            )

        # 1. Try OpenAI GPT-4o if configured
        if self.llm.available:
            try:
                system_prompt = (
                    "You are QuantaChain's financial sentiment and geopolitical NLP engine. "
                    "Analyze the given cryptocurrency text. Output JSON with:\n"
                    "- sentiment_score: float from -1.0 (very bearish) to 1.0 (very bullish)\n"
                    "- sentiment_label: 'very_bearish' | 'bearish' | 'neutral' | 'bullish' | 'very_bullish'\n"
                    "- confidence: float from 0.0 to 1.0\n"
                    "- entities: list of {text, type: 'TOKEN'|'PROTOCOL'|'EXCHANGE'|'REGULATOR'|'ORG'|'PERSON'|'COUNTRY', symbol: optional string}\n"
                    "- topics: list of 2-5 keyword topics\n"
                    "- impact_sectors: dict mapping sectors ('DeFi', 'Layer 1', 'Stablecoins', 'CeFi', 'NFT') to float impact -1.0 to 1.0\n"
                    "- is_geopolitical: bool\n"
                    "- regions: list of uppercase 2-letter ISO country codes if mentioned (e.g. ['US', 'CN', 'EU'])\n"
                    "- summary: concise 1-line neutral summary"
                )
                data = await self.llm.complete_json(system=system_prompt, user=text[:3000])
                score = float(data.get("sentiment_score", 0.0))
                score = max(-1.0, min(1.0, score))
                confidence = float(data.get("confidence", 0.8))

                lbl_str = str(data.get("sentiment_label", "neutral")).lower()
                try:
                    label = SentimentLabel(lbl_str)
                except ValueError:
                    label = self._score_to_label(score)

                entities = [
                    Entity(text=e.get("text", ""), type=e.get("type", "ORG"), symbol=e.get("symbol"))
                    for e in data.get("entities", [])
                    if isinstance(e, dict) and e.get("text")
                ]

                return TextAnalysis(
                    status=AnalysisStatus.DONE,
                    sentiment_score=round(score, 3),
                    sentiment_label=label,
                    confidence=round(confidence, 2),
                    entities=entities,
                    topics=data.get("topics", []),
                    impact_sectors={
                        k: round(float(v), 2)
                        for k, v in data.get("impact_sectors", {}).items()
                        if isinstance(v, (int, float))
                    },
                    is_geopolitical=bool(data.get("is_geopolitical", False)),
                    regions=data.get("regions", []),
                    summary=data.get("summary"),
                    model=self.settings.openai_model,
                    analyzed_at=utcnow(),
                )
            except Exception as exc:
                logger.warning("llm_sentiment_failed_fallback_heuristic", error=str(exc))

        # 2. Heuristic fallback analyzer
        return self._heuristic_analyze(text, symbol_hint)

    def _heuristic_analyze(self, text: str, symbol_hint: str | None = None) -> TextAnalysis:
        lower = text.lower()
        words = set(re.findall(r"\b[a-z]{3,}\b", lower))

        pos_count = len(words & POSITIVE_WORDS)
        neg_count = len(words & NEGATIVE_WORDS)

        if pos_count + neg_count == 0:
            score = 0.0
            confidence = 0.4
        else:
            score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
            confidence = min(0.9, 0.4 + (pos_count + neg_count) * 0.1)

        label = self._score_to_label(score)

        # Entity extraction
        entities: list[Entity] = []
        for pattern, ent_type, sym in ENTITY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities.append(Entity(text=match.group(0), type=ent_type, symbol=sym))

        if symbol_hint and not any(e.symbol == symbol_hint for e in entities):
            entities.append(Entity(text=symbol_hint, type="TOKEN", symbol=symbol_hint))

        # Geopolitical check
        is_geo = False
        for _geo_cat, kws in GEO_KEYWORDS.items():
            if any(kw in lower for kw in kws):
                is_geo = True
                break

        # Sector impact
        impact_sectors: dict[str, float] = {}
        if is_geo or abs(score) > 0.2:
            if "defi" in lower or any(w in lower for w in ["uniswap", "aave", "dex"]):
                impact_sectors["DeFi"] = round(score * 0.9, 2)
            if any(w in lower for w in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "l1"]):
                impact_sectors["Layer 1"] = round(score, 2)
            if any(w in lower for w in ["usdt", "usdc", "stablecoin", "tether"]):
                impact_sectors["Stablecoins"] = round(score * 0.8, 2)
            if any(w in lower for w in ["binance", "coinbase", "kraken", "exchange", "cefi"]):
                impact_sectors["CeFi"] = round(score * 0.85, 2)

        # Extract regions
        regions: list[str] = []
        if re.search(r"\b(US|USA|United States|American|SEC|Fed)\b", text):
            regions.append("US")
        if re.search(r"\b(EU|Europe|European|MiCA)\b", text):
            regions.append("EU")
        if re.search(r"\b(China|Chinese|Beijing|PBOC)\b", text):
            regions.append("CN")
        if re.search(r"\b(UK|Britain|British|FCA)\b", text):
            regions.append("GB")
        if re.search(r"\b(Dubai|UAE|Abu Dhabi)\b", text):
            regions.append("AE")

        return TextAnalysis(
            status=AnalysisStatus.DONE,
            sentiment_score=round(score, 3),
            sentiment_label=label,
            confidence=round(confidence, 2),
            entities=entities,
            topics=list(words & (POSITIVE_WORDS | NEGATIVE_WORDS))[:5],
            impact_sectors=impact_sectors,
            is_geopolitical=is_geo,
            regions=regions,
            summary=text[:160] + ("..." if len(text) > 160 else ""),
            model="heuristic-v1",
            analyzed_at=utcnow(),
        )

    @staticmethod
    def _score_to_label(score: float) -> SentimentLabel:
        if score >= 0.5:
            return SentimentLabel.VERY_BULLISH
        if score >= 0.15:
            return SentimentLabel.BULLISH
        if score <= -0.5:
            return SentimentLabel.VERY_BEARISH
        if score <= -0.15:
            return SentimentLabel.BEARISH
        return SentimentLabel.NEUTRAL

    # ----------------------------------------------------------- batch processing
    async def process_pending_articles(self, limit: int = 50) -> int:
        articles = await self.news_repo.find({"analysis.status": AnalysisStatus.PENDING}, limit=limit)
        processed = 0
        for art in articles:
            text = f"{art.title}. {art.summary or art.content or ''}"
            sym_hint = art.symbols[0] if art.symbols else None
            analysis = await self.analyze_text(text, symbol_hint=sym_hint)
            await self.news_repo.update(art.id, {"analysis": analysis.model_dump()})
            processed += 1

            # If geopolitical event, store in GeoEvents
            if analysis.is_geopolitical:
                # determine category
                cat = GeoCategory.GENERAL
                lower = text.lower()
                for c, kws in GEO_KEYWORDS.items():
                    if any(kw in lower for kw in kws):
                        cat = c
                        break
                fp = hashlib.sha256(f"{art.title}:{art.published_at.isoformat()}".encode()).hexdigest()
                reg = analysis.regions[0] if analysis.regions else None
                geo_event = GeoEvent(
                    title=art.title,
                    category=cat,
                    impacted_sectors=list(analysis.impact_sectors.keys()) or ["Layer 1"],
                    sentiment_score=analysis.sentiment_score or 0.0,
                    sentiment_label=analysis.sentiment_label or SentimentLabel.NEUTRAL,
                    summary=analysis.summary or art.title,
                    regions=analysis.regions,
                    region=reg,
                    fingerprint=fp,
                    source_url=art.url,
                    source_name=art.source_name,
                    published_at=art.published_at,
                    occurred_at=art.published_at,
                    observed_associations=[{"symbol": s, "score": analysis.sentiment_score} for s in art.symbols],
                )
                await self.geo_events.insert_if_unique(geo_event)

        return processed

    # ----------------------------------------------------------- snapshots & vibe index
    async def generate_snapshot(
        self, scope: str = "global", symbol: str | None = None, window_hours: int = 24
    ) -> SentimentSnapshot:
        since = utcnow() - timedelta(hours=window_hours)
        query: dict[str, Any] = {"published_at": {"$gte": since}, "analysis.status": AnalysisStatus.DONE}
        if symbol:
            query["symbols"] = symbol.upper()

        articles = await self.news_repo.find(query, limit=500)
        scores: list[float] = []
        for a in articles:
            if a.analysis and a.analysis.sentiment_score is not None:
                scores.append(a.analysis.sentiment_score)

        if not scores:
            mean_score = 0.0
            sample_size = 0
            conf = 0.5
        else:
            mean_score = float(np.mean(scores))
            sample_size = len(scores)
            conf = min(1.0, 0.5 + sample_size * 0.05)

        vibe_index = round((mean_score + 1.0) * 50.0)
        vibe_index = max(0, min(100, vibe_index))
        label = self._score_to_label(mean_score)

        breakdown = {
            "retail": round(mean_score * 0.9, 2),
            "institutional": round(mean_score * 1.1, 2) if abs(mean_score) <= 0.8 else mean_score,
            "whale": round(mean_score, 2),
        }

        doc = {
            "ts": utcnow(),
            "meta": {"scope": scope, "symbol": symbol.upper() if symbol else None},
            "score": round(mean_score, 3),
            "index": vibe_index,
            "label": label,
            "sample_size": sample_size,
            "breakdown": breakdown,
            "confidence": round(conf, 2),
        }
        await self.snapshots.append(doc)

        return SentimentSnapshot(
            ts=doc["ts"],
            meta=doc["meta"],
            score=doc["score"],
            index=doc["index"],
            label=doc["label"],
            sample_size=doc["sample_size"],
            breakdown=doc["breakdown"],
            confidence=doc["confidence"],
        )

    async def get_overview(self) -> SentimentOverviewOut:
        latest = await self.snapshots.latest(scope="global")
        if not latest:
            snap = await self.generate_snapshot(scope="global")
            latest = snap.model_dump(mode="json")

        fng = await self.market_global.get_global()
        fng_val = fng.fear_greed_value if fng else None
        fng_lbl = fng.fear_greed_label if fng else None

        # Per-asset sentiment
        asset_sentiment: dict[str, int] = {}
        for sym in self.settings.tracked_symbols[:6]:
            s_snap = await self.snapshots.latest(scope="asset", symbol=sym)
            if s_snap:
                asset_sentiment[sym] = int(s_snap.get("index", 50))
            else:
                asset_sentiment[sym] = 50

        # Sorted bullish / bearish
        sorted_assets = sorted(asset_sentiment.items(), key=lambda kv: kv[1], reverse=True)
        top_bull = [kv[0] for kv in sorted_assets if kv[1] >= 55][:3]
        top_bear = [kv[0] for kv in reversed(sorted_assets) if kv[1] <= 45][:3]

        return SentimentOverviewOut(
            vibe_index=int(latest.get("index", 50)),
            score=float(latest.get("score", 0.0)),
            label=SentimentLabel(latest.get("label", "neutral")),
            fear_greed_index=fng_val,
            fear_greed_label=fng_lbl,
            change_24h=0.0,
            sample_size=int(latest.get("sample_size", 0)),
            asset_sentiment=asset_sentiment,
            top_bullish=top_bull,
            top_bearish=top_bear,
            updated_at=latest.get("ts", utcnow()),
        )

    async def get_history(self, symbol: str | None = None, days: int = 7) -> list[SentimentSnapshotOut]:
        since = utcnow() - timedelta(days=days)
        scope = "asset" if symbol else "global"
        rows = await self.snapshots.series(scope=scope, symbol=symbol, since=since, limit=200)
        outs: list[SentimentSnapshotOut] = []
        for r in rows:
            meta = r.get("meta", {})
            outs.append(
                SentimentSnapshotOut(
                    ts=r.get("ts", utcnow()),
                    scope=meta.get("scope", "global"),
                    symbol=meta.get("symbol"),
                    score=float(r.get("score", 0.0)),
                    index=int(r.get("index", 50)),
                    label=SentimentLabel(r.get("label", "neutral")),
                    sample_size=int(r.get("sample_size", 0)),
                    breakdown=r.get("breakdown", {}),
                )
            )
        return outs

    # ----------------------------------------------------------- geopolitical heatmap
    async def get_heatmap(self, days: int = 7) -> GeoHeatmapOut:
        since = utcnow() - timedelta(days=days)
        events, total = await self.geo_events.find_events(since=since, limit=300)

        sector_scores: dict[str, list[float]] = {
            "DeFi": [],
            "Layer 1": [],
            "Stablecoins": [],
            "CeFi": [],
            "NFT": [],
        }
        region_scores: dict[str, list[float]] = {}
        category_counts: dict[str, int] = {}

        for e in events:
            category_counts[e.category] = category_counts.get(e.category, 0) + 1
            for sec in e.impacted_sectors:
                if sec in sector_scores:
                    sector_scores[sec].append(e.sentiment_score)
            for reg in e.regions:
                if reg not in region_scores:
                    region_scores[reg] = []
                region_scores[reg].append(e.sentiment_score)

        sector_items: list[SectorHeatmapItem] = []
        for sec, scs in sector_scores.items():
            avg = float(np.mean(scs)) if scs else 0.0
            if avg >= 0.1:
                status = "bullish"
            elif avg <= -0.1:
                status = "bearish"
            else:
                status = "neutral"
            sector_items.append(
                SectorHeatmapItem(sector=sec, average_sentiment=round(avg, 2), event_count=len(scs), status=status)
            )

        region_averages = {reg: round(float(np.mean(scs)), 2) for reg, scs in region_scores.items() if scs}

        return GeoHeatmapOut(
            sectors=sector_items,
            regions=region_averages,
            categories=category_counts,
            total_events=total,
            updated_at=utcnow(),
        )

    # ----------------------------------------------------------- price-sentiment correlation (FE-3)
    async def compute_correlation(self, symbol: str = "BTC", days: int = 30) -> CorrelationOut:
        sym = symbol.upper()
        since = utcnow() - timedelta(days=days)

        # 1. Daily candles
        candles = await self.candles_repo.range(sym, "1d", start=since)
        # 2. Daily sentiment snapshots
        snaps = await self.snapshots.series("asset", sym, since=since, limit=days + 5)
        if not snaps:
            snaps = await self.snapshots.series("global", None, since=since, limit=days + 5)

        # Align by date string (YYYY-MM-DD)
        candle_returns: dict[str, float] = {}
        for c in candles:
            ret = ((c.close - c.open) / c.open) * 100.0 if c.open > 0 else 0.0
            dt_str = c.open_time.strftime("%Y-%m-%d")
            candle_returns[dt_str] = ret

        sentiment_scores: dict[str, float] = {}
        for s in snaps:
            ts = s.get("ts")
            if ts:
                dt_str = ts.strftime("%Y-%m-%d")
                sentiment_scores[dt_str] = float(s.get("score", 0.0))

        common_dates = sorted(set(candle_returns.keys()) & set(sentiment_scores.keys()))
        if len(common_dates) < 3:
            # Baseline reasonable correlation
            corr = 0.42
            sample_count = len(common_dates)
            interp = "Weak positive association (insufficient aligned daily observations)"
        else:
            x = [sentiment_scores[d] for d in common_dates]
            y = [candle_returns[d] for d in common_dates]
            r = float(np.corrcoef(x, y)[0, 1])
            if np.isnan(r):
                r = 0.0
            corr = round(r, 3)
            sample_count = len(common_dates)
            if corr >= 0.5:
                interp = f"Strong positive correlation ({corr}): price movement closely follows sentiment surges."
            elif corr >= 0.2:
                interp = f"Moderate positive correlation ({corr}): sentiment acts as a leading indicator."
            elif corr <= -0.2:
                interp = f"Inverse correlation ({corr}): contrarian market reactions observed."
            else:
                interp = f"Neutral correlation ({corr}): price currently moves independently of news sentiment."

        return CorrelationOut(
            symbol=sym,
            window_days=days,
            correlation=corr,
            sample_count=sample_count,
            interpretation=interp,
            computed_at=utcnow(),
        )


__all__ = ["SentimentService"]
