"""Seed a local database with demo data so every dashboard has something to show.

    python -m scripts.seed_demo                      # synthetic candles + news + admin user
    python -m scripts.seed_demo --no-synthetic       # only assets, news samples and the admin user

Live ingestion (worker) will overwrite the synthetic market data with real data
as it arrives; the synthetic candles are tagged ``source="synthetic"``.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.timeutils import utcnow
from app.db.collections import ensure_schema
from app.db.mongo import mongo
from app.integrations.news import content_hash, url_hash
from app.modules.market.models import NewsArticle, SocialPost
from app.modules.market.repository import AssetRepository, MarketGlobalRepository, NewsRepository, SocialRepository
from app.modules.market.seed import ensure_default_assets, seed_synthetic_market
from app.modules.market.symbols import SymbolDetector
from scripts.create_admin import create_admin

SAMPLE_HEADLINES: list[tuple[str, str, str]] = [
    (
        "SEC approves options trading on spot Bitcoin ETFs",
        "Regulators cleared the way for derivatives on BTC ETFs, a move analysts call bullish for institutional adoption.",
        "CoinDesk",
    ),
    (
        "Ethereum developers finalise next upgrade timeline as staking yields compress",
        "Core devs agreed on the roadmap; liquid restaking tokens continue to gain share.",
        "The Block",
    ),
    (
        "Federal Reserve holds rates steady, signals patience on cuts",
        "Risk assets including crypto slipped after the announcement before recovering.",
        "Reuters",
    ),
    (
        "Solana network activity hits record as memecoin trading surges",
        "Daily active addresses climbed to a new high while fees remained low.",
        "Decrypt",
    ),
    (
        "Hackers drain $40M from DeFi lending protocol in flash-loan exploit",
        "The team paused the contracts and is negotiating with the attacker.",
        "Cointelegraph",
    ),
    (
        "EU MiCA rules come into force for stablecoin issuers",
        "Exchanges began delisting non-compliant euro stablecoins across the bloc.",
        "CoinDesk",
    ),
    (
        "Binance reports record inflows as traders rotate into large caps",
        "Exchange netflows turned positive for BTC and ETH this week.",
        "The Block",
    ),
    (
        "Pakistan explores regulatory framework for digital assets",
        "Officials discussed licensing exchanges and taxing crypto gains.",
        "Dawn",
    ),
    (
        "Dogecoin jumps 12% after social media frenzy",
        "Retail traders piled in while whales distributed into strength.",
        "Decrypt",
    ),
    (
        "Chainlink expands cross-chain interoperability protocol to new banks",
        "CCIP pilots with major financial institutions signal growing enterprise demand.",
        "Cointelegraph",
    ),
]

SAMPLE_POSTS: list[tuple[str, str, int]] = [
    (
        "BTC breaking out, this is the most bullish structure in months",
        "Volume confirming, ETF flows strong. Loading up.",
        420,
    ),
    (
        "Why I'm bearish on alts until ETH reclaims 4k",
        "Rotation isn't happening, liquidity is thin, be careful out there.",
        150,
    ),
    ("SOL fees are insane right now (in a good way)", "Network handling the memecoin madness fine.", 300),
    (
        "Rug pulled again... never aping into unaudited contracts",
        "Liquidity got yanked 10 minutes after launch. Lesson learned.",
        90,
    ),
    ("DOGE to the moon?", "Elon tweeted again lol", 60),
]


async def seed_content(db) -> dict[str, int]:  # type: ignore[no-untyped-def]
    detector = SymbolDetector(await AssetRepository(db).tracked())
    news_repo, social_repo = NewsRepository(db), SocialRepository(db)
    now = utcnow()
    inserted = {"news": 0, "social": 0}
    for index, (title, summary, source) in enumerate(SAMPLE_HEADLINES):
        url = f"https://example.org/demo/{index}"
        article = NewsArticle(
            url=url,
            url_hash=url_hash(url),
            title=title,
            summary=summary,
            source_name=source,
            origin="rss",
            published_at=now - timedelta(hours=2 * index + 1),
            fetched_at=now,
            symbols=detector.detect(title, summary),
        )
        inserted["news"] += int(await news_repo.insert_if_new(article))
    for index, (title, text, score) in enumerate(SAMPLE_POSTS):
        post = SocialPost(
            platform="reddit",
            external_id=f"demo:{index}",
            content_hash=content_hash(f"{title} {text}"),
            title=title,
            text=text,
            url=f"https://reddit.com/r/CryptoCurrency/demo{index}",
            author="demo_user",
            posted_at=now - timedelta(hours=index + 1),
            score=score,
            comments=score // 10,
            community="CryptoCurrency",
            symbols=detector.detect(title, text),
        )
        inserted["social"] += int(await social_repo.insert_if_new(post))
    return inserted


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Quantachain demo data")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic candles / market stats")
    parser.add_argument("--admin-email", default="admin@quantachain.local")
    parser.add_argument("--admin-password", default="Admin1234!")
    args = parser.parse_args()

    configure_logging("WARNING")
    settings = get_settings()
    db = await mongo.connect(settings)
    await ensure_schema(db, settings)
    created_assets = await ensure_default_assets(db, settings)
    print(f"assets ensured ({created_assets} created)")

    if not args.no_synthetic:
        written = await seed_synthetic_market(
            db, settings.tracked_symbols, intervals=("1m", "5m", "15m", "1h", "4h", "1d")
        )
        print(f"synthetic candles written: {written}")
        global_repo = MarketGlobalRepository(db)
        if await global_repo.get_global() is None:
            await global_repo.update_global(
                total_market_cap_usd=2.42e12,
                total_volume_24h_usd=84.2e9,
                btc_dominance_pct=52.4,
                eth_dominance_pct=17.1,
                market_cap_change_24h_pct=1.2,
                active_cryptocurrencies=14_000,
                fear_greed_value=72,
                fear_greed_label="Greed",
                fear_greed_updated_at=utcnow(),
            )
            print("market overview seeded (synthetic)")

    content = await seed_content(db)
    print(f"sample content inserted: {content}")

    await mongo.close()
    action = await create_admin(args.admin_email, args.admin_password, "Demo Admin")
    print(f"admin {action}: {args.admin_email} / {args.admin_password}")
    print("Start the worker (python -m app.worker) to replace synthetic data with live feeds.")


if __name__ == "__main__":
    asyncio.run(main())
