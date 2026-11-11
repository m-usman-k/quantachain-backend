"""Seed a local database with demo data so every dashboard has something to show.

    python -m scripts.seed_demo                      # synthetic candles + news + admin user
    python -m scripts.seed_demo --no-synthetic       # only assets, news samples and the admin user

Live ingestion (worker) will overwrite the synthetic market data with real data
as it arrives; the synthetic candles are tagged ``source="synthetic"``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import timedelta

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.timeutils import utcnow
from app.db.collections import ensure_schema
from app.db.mongo import mongo
from app.integrations.news import content_hash, url_hash
from app.modules.fraud.models import (
    AlertSeverity,
    AlertType,
    BlacklistEntry,
    BlacklistSource,
    BlacklistStatus,
    FraudAlert,
    ProjectScore,
    ScoreComponent,
    ScoreLabel,
    WatchedPair,
)
from app.modules.fraud.repository import (
    BlacklistRepository,
    FraudAlertRepository,
    ProjectScoreRepository,
    WatchedPairRepository,
)
from app.modules.market.models import NewsArticle, SentimentLabel, SocialPost
from app.modules.market.repository import AssetRepository, MarketGlobalRepository, NewsRepository, SocialRepository
from app.modules.market.seed import ensure_default_assets, seed_synthetic_market
from app.modules.market.symbols import SymbolDetector
from app.modules.onchain.models import ConfirmationStatus, EntityType, TransferFlow, TransferRisk, Wallet, WhaleTransfer
from app.modules.onchain.repository import WalletRepository, WhaleTransferRepository
from app.modules.sentiment.models import GeoCategory, GeoEvent, SentimentSnapshot
from app.modules.sentiment.repository import GeoEventRepository, SentimentSnapshotRepository
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


async def seed_onchain_demo(db) -> dict[str, int]:  # type: ignore[no-untyped-def]
    wallet_repo = WalletRepository(db)
    whale_repo = WhaleTransferRepository(db)
    now = utcnow()
    stats = {"wallets": 0, "transfers": 0}

    wallets = [
        Wallet(
            address="0x28c6c06298d514db089934071355e5743bf21d60",
            chain="eth",
            label="Binance 14 Hot Wallet",
            entity_type=EntityType.EXCHANGE,
            balance_usd=1_450_000_000.0,
            smart_money_score=40,
            tags=["exchange", "hot-wallet", "tier1"],
            is_exchange=True,
            first_seen_at=now - timedelta(days=365),
            last_seen_at=now,
        ),
        Wallet(
            address="0x94845333028b1204fbe13e12b05369a33c88e5d0",
            chain="eth",
            label="Jump Trading",
            entity_type=EntityType.SMART_MONEY,
            balance_usd=420_000_000.0,
            smart_money_score=92,
            win_rate=0.74,
            total_profit_usd=45_200_000.0,
            tags=["smart-money", "market-maker", "fund"],
            first_seen_at=now - timedelta(days=400),
            last_seen_at=now,
        ),
        Wallet(
            address="0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
            chain="eth",
            label="vitalik.eth",
            entity_type=EntityType.INDIVIDUAL,
            balance_usd=320_000_000.0,
            smart_money_score=85,
            tags=["founder", "whale", "notable"],
            first_seen_at=now - timedelta(days=1200),
            last_seen_at=now,
        ),
        Wallet(
            address="0x8589427373d6d84e98730d7795d8f6f8731fda16",
            chain="eth",
            label="Tornado Cash Router",
            entity_type=EntityType.CONTRACT,
            balance_usd=12_000_000.0,
            smart_money_score=10,
            tags=["mixer", "sanctioned", "privacy"],
            first_seen_at=now - timedelta(days=500),
            last_seen_at=now,
        ),
        Wallet(
            address="0x503828976d22510aad0201ac7ec88293211d23dc",
            chain="eth",
            label="Wintermute Trading",
            entity_type=EntityType.SMART_MONEY,
            balance_usd=185_000_000.0,
            smart_money_score=88,
            win_rate=0.69,
            total_profit_usd=28_400_000.0,
            tags=["market-maker", "algorithmic"],
            first_seen_at=now - timedelta(days=600),
            last_seen_at=now,
        ),
    ]

    for w in wallets:
        existing = await wallet_repo.get_by_address(w.address)
        if existing is None:
            await wallet_repo.insert(w)
            stats["wallets"] += 1

    transfers = [
        WhaleTransfer(
            tx_hash="0xaa10000000000000000000000000000000000000000000000000000000000001",
            log_index=0,
            chain="eth",
            block_number=19800001,
            block_time=now - timedelta(minutes=45),
            symbol="ETH",
            amount=4500.0,
            amount_usd=15_750_000.0,
            price_source="synthetic",
            price_timestamp=now - timedelta(minutes=45),
            from_address="0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
            to_address="0x28c6c06298d514db089934071355e5743bf21d60",
            from_label="vitalik.eth",
            to_label="Binance 14 Hot Wallet",
            flow=TransferFlow.EXCHANGE_INFLOW,
            risk_level=TransferRisk.LOW,
            confirmation_status=ConfirmationStatus.CONFIRMED,
            gas_fee_eth=0.0042,
        ),
        WhaleTransfer(
            tx_hash="0xaa10000000000000000000000000000000000000000000000000000000000002",
            log_index=0,
            chain="eth",
            block_number=19800050,
            block_time=now - timedelta(hours=2),
            symbol="USDT",
            amount=25_000_000.0,
            amount_usd=25_000_000.0,
            price_source="synthetic",
            price_timestamp=now - timedelta(hours=2),
            from_address="0x28c6c06298d514db089934071355e5743bf21d60",
            to_address="0x503828976d22510aad0201ac7ec88293211d23dc",
            from_label="Binance 14 Hot Wallet",
            to_label="Wintermute Trading",
            flow=TransferFlow.EXCHANGE_OUTFLOW,
            risk_level=TransferRisk.LOW,
            confirmation_status=ConfirmationStatus.CONFIRMED,
            gas_fee_eth=0.0035,
        ),
        WhaleTransfer(
            tx_hash="0xaa10000000000000000000000000000000000000000000000000000000000003",
            log_index=0,
            chain="eth",
            block_number=19800100,
            block_time=now - timedelta(hours=4),
            symbol="ETH",
            amount=1200.0,
            amount_usd=4_200_000.0,
            price_source="synthetic",
            price_timestamp=now - timedelta(hours=4),
            from_address="0x94845333028b1204fbe13e12b05369a33c88e5d0",
            to_address="0x8589427373d6d84e98730d7795d8f6f8731fda16",
            from_label="Jump Trading",
            to_label="Tornado Cash Router",
            flow=TransferFlow.WALLET_TO_WALLET,
            risk_level=TransferRisk.SUSPICIOUS,
            confirmation_status=ConfirmationStatus.CONFIRMED,
            gas_fee_eth=0.015,
        ),
        WhaleTransfer(
            tx_hash="0xaa10000000000000000000000000000000000000000000000000000000000004",
            log_index=0,
            chain="eth",
            block_number=19800150,
            block_time=now - timedelta(hours=6),
            symbol="BTC",
            amount=350.0,
            amount_usd=23_800_000.0,
            price_source="synthetic",
            price_timestamp=now - timedelta(hours=6),
            from_address="0x503828976d22510aad0201ac7ec88293211d23dc",
            to_address="0x94845333028b1204fbe13e12b05369a33c88e5d0",
            from_label="Wintermute Trading",
            to_label="Jump Trading",
            flow=TransferFlow.WALLET_TO_WALLET,
            risk_level=TransferRisk.LOW,
            confirmation_status=ConfirmationStatus.CONFIRMED,
            gas_fee_eth=0.005,
        ),
    ]

    for t in transfers:
        if await whale_repo.insert_transfer(t):
            stats["transfers"] += 1

    return stats


async def seed_sentiment_demo(db) -> dict[str, int]:  # type: ignore[no-untyped-def]
    snap_repo = SentimentSnapshotRepository(db)
    geo_repo = GeoEventRepository(db)
    now = utcnow()
    stats = {"snapshots": 0, "geo_events": 0}

    # Seed 24 hourly snapshots for global market and BTC
    for h in range(24, -1, -1):
        ts = now - timedelta(hours=h)
        wave = ((h % 6) - 3) / 10.0
        score = max(-1.0, min(1.0, 0.42 + wave))
        vibe_idx = int(50 + score * 45)
        label = (
            SentimentLabel.BULLISH
            if score > 0.15
            else (SentimentLabel.BEARISH if score < -0.15 else SentimentLabel.NEUTRAL)
        )
        snap = SentimentSnapshot(
            ts=ts,
            meta={"scope": "global"},
            score=round(score, 3),
            index=vibe_idx,
            label=label,
            sample_size=120 + h * 4,
            breakdown={"retail": 0.45, "institutional": 0.35, "whale": 0.20},
            confidence=0.92,
        )
        await snap_repo.append(snap.to_mongo())
        stats["snapshots"] += 1

        btc_snap = SentimentSnapshot(
            ts=ts,
            meta={"scope": "asset", "symbol": "BTC"},
            score=round(score * 1.1, 3),
            index=min(100, max(0, int(vibe_idx * 1.05))),
            label=label,
            sample_size=80 + h * 2,
            breakdown={"retail": 0.50, "institutional": 0.30, "whale": 0.20},
            confidence=0.94,
        )
        await snap_repo.append(btc_snap.to_mongo())
        stats["snapshots"] += 1

    geo_events = [
        (
            "US SEC Approves Options Trading on Spot Bitcoin ETFs",
            GeoCategory.REGULATION,
            ["DeFi", "CeFi"],
            0.75,
            SentimentLabel.BULLISH,
            "Regulators authorized options trading on spot BTC ETF instruments, unlocking institutional risk management tools.",
            ["US"],
            "US",
            now - timedelta(hours=3),
        ),
        (
            "European Union MiCA Regulatory Framework Final Implementation Phase Commences",
            GeoCategory.REGULATION,
            ["Stablecoins", "CeFi"],
            0.15,
            SentimentLabel.NEUTRAL,
            "Markets prepare for comprehensive stablecoin compliance across European banking corridors.",
            ["EU"],
            "EU",
            now - timedelta(hours=8),
        ),
        (
            "Federal Reserve Signals Measured Pace on Benchmark Rate Adjustments",
            GeoCategory.MACRO,
            ["Layer 1", "DeFi"],
            -0.35,
            SentimentLabel.BEARISH,
            "Central bank commentary suggests higher-for-longer yields, temporarily curbing risk-on capital rotation.",
            ["US"],
            "US",
            now - timedelta(hours=14),
        ),
        (
            "UAE Financial Services Regulatory Authority Expands Digital Asset Broker Licenses",
            GeoCategory.REGULATION,
            ["DeFi", "Layer 1"],
            0.65,
            SentimentLabel.BULLISH,
            "Abu Dhabi Global Market registers international institutional custodians and decentralized protocols.",
            ["AE"],
            "AE",
            now - timedelta(hours=22),
        ),
    ]

    for title, cat, sectors, score, label, summary, regions, region, dt in geo_events:
        fp = hashlib.sha256(f"{title}{dt.date()}".encode()).hexdigest()
        event = GeoEvent(
            title=title,
            category=cat,
            impacted_sectors=sectors,
            sentiment_score=score,
            sentiment_label=label,
            summary=summary,
            regions=regions,
            region=region,
            fingerprint=fp,
            source_name="Market Intelligence",
            published_at=dt,
            occurred_at=dt,
        )
        inserted = await geo_repo.insert_if_unique(event)
        if inserted:
            stats["geo_events"] += 1

    return stats


async def seed_fraud_demo(db) -> dict[str, int]:  # type: ignore[no-untyped-def]
    bl_repo = BlacklistRepository(db)
    pair_repo = WatchedPairRepository(db)
    alert_repo = FraudAlertRepository(db)
    score_repo = ProjectScoreRepository(db)
    now = utcnow()
    stats = {"blacklist": 0, "pairs": 0, "alerts": 0, "scores": 0}

    blacklist_entries = [
        BlacklistEntry(
            address="0x8589427373d6d84e98730d7795d8f6f8731fda16",
            chain="eth",
            risk_score=95,
            reason="OFAC Sanctioned mixer contract (Tornado Cash Router)",
            status=BlacklistStatus.BLOCKED,
            source=BlacklistSource.MANUAL,
            tags=["mixer", "sanctioned", "ofac"],
            notes="Flagged under international AML sanction guidelines.",
        ),
        BlacklistEntry(
            address="0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
            chain="eth",
            risk_score=99,
            reason="Euler Finance flash-loan exploit drained wallet",
            status=BlacklistStatus.BLOCKED,
            source=BlacklistSource.MITIGATION,
            tags=["exploit", "drainer"],
            notes="Involved in high-profile protocol exploit.",
        ),
    ]
    for b in blacklist_entries:
        existing = await bl_repo.get_by_address(b.address)
        if existing is None:
            await bl_repo.insert(b)
            stats["blacklist"] += 1

    pairs = [
        WatchedPair(
            chain="ethereum",
            pair_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
            token_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            base_symbol="USDC",
            quote_symbol="WETH",
            dex="uniswap-v3",
            label="Uniswap v3 USDC/WETH pool",
            last_liquidity_usd=120_000_000.0,
            last_price_usd=1.0,
            last_checked_at=now,
        ),
        WatchedPair(
            chain="ethereum",
            pair_address="0xa43fe1690fb5055c6139446d338f090d81373507",
            token_address="0x6982508145454ce325ddbe47a25d4ec3d2311933",
            base_symbol="PEPE",
            quote_symbol="WETH",
            dex="uniswap-v2",
            label="PEPE / WETH Uniswap v2",
            last_liquidity_usd=15_000_000.0,
            last_price_usd=0.000018,
            last_checked_at=now,
        ),
    ]
    for p in pairs:
        existing = await pair_repo.get_pair(p.chain, p.pair_address)
        if existing is None:
            await pair_repo.insert(p)
            stats["pairs"] += 1

    alerts = [
        FraudAlert(
            type=AlertType.RUG_PULL,
            severity=AlertSeverity.CRITICAL,
            token_address="0x000000000000000000000000000000000000dead",
            pair_address="0x000000000000000000000000000000000000dead",
            chain="ethereum",
            symbol="FAKETOKEN",
            title="Abnormal 95% Liquidity Drain Detected",
            body="Uniswap pool liquidity removed within 2 blocks without timelock restriction.",
            evidence={"drain_pct": 95.2, "dex": "uniswap-v2", "withdrawn_eth": 120.0},
            fingerprint=hashlib.sha256(b"alert-rugpull-demo-001").hexdigest(),
            risk_score=94,
        ),
        FraudAlert(
            type=AlertType.CONTRACT_VULNERABILITY,
            severity=AlertSeverity.WARNING,
            token_address="0x1111111111111111111111111111111111111111",
            chain="ethereum",
            symbol="VULN",
            title="Reentrancy Vulnerability Detected in withdraw()",
            body="Static analysis identified an external call before state balance reset (SWC-107).",
            evidence={"swc": "SWC-107", "function": "withdraw", "severity": "HIGH"},
            fingerprint=hashlib.sha256(b"alert-vuln-demo-002").hexdigest(),
            risk_score=78,
        ),
    ]
    for a in alerts:
        existing = await alert_repo.find_one({"fingerprint": a.fingerprint})
        if existing is None:
            await alert_repo.insert(a)
            stats["alerts"] += 1

    existing_score = await score_repo.get_for("eth", "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984")
    if existing_score is None:
        proj_score = ProjectScore(
            chain="eth",
            address="0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            symbol="UNI",
            score=88,
            label=ScoreLabel.EXCELLENT,
            components=[
                ScoreComponent(
                    name="Contract Security", score=28, max=30, detail="Verified bytecode and public audits"
                ),
                ScoreComponent(name="Liquidity Health", score=24, max=25, detail="Deep DEX liquidity on Uniswap v3"),
                ScoreComponent(
                    name="Holder Distribution", score=18, max=20, detail="Well-distributed governance token"
                ),
                ScoreComponent(
                    name="Governance Transparency", score=10, max=15, detail="Decentralized on-chain voting active"
                ),
                ScoreComponent(
                    name="Market Volatility", score=8, max=10, detail="Stable trading volume across major venues"
                ),
            ],
            inputs={"dex_liquidity_usd": 150_000_000.0, "scanned_vulnerabilities": 0},
            computed_at=now,
        )
        await score_repo.insert(proj_score)
        stats["scores"] += 1

    return stats


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

    onchain_res = await seed_onchain_demo(db)
    print(f"on-chain demo data inserted: {onchain_res}")

    sentiment_res = await seed_sentiment_demo(db)
    print(f"sentiment demo data inserted: {sentiment_res}")

    fraud_res = await seed_fraud_demo(db)
    print(f"fraud & risk demo data inserted: {fraud_res}")

    await mongo.close()
    action = await create_admin(args.admin_email, args.admin_password, "Demo Admin")
    print(f"admin {action}: {args.admin_email} / {args.admin_password}")
    print("Start the worker (python -m app.worker) to replace synthetic data with live feeds.")


if __name__ == "__main__":
    asyncio.run(main())
