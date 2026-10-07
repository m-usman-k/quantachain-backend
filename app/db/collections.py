"""Collection names, indexes and time-series definitions.

``ensure_schema`` is idempotent and runs on every start-up (API and worker), so
a fresh MongoDB is fully prepared without a separate migration step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid, OperationFailure

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class Collections:
    # module 1 - auth
    USERS = "users"
    REFRESH_TOKENS = "refresh_tokens"
    AUDIT_LOGS = "audit_logs"
    API_KEYS = "api_keys"
    # notifications
    DEVICE_TOKENS = "device_tokens"
    NOTIFICATIONS = "notifications"
    # module 2 - market data
    ASSETS = "assets"
    PRICE_TICKS = "price_ticks"  # time-series
    CANDLES = "candles"
    ASSET_STATS = "asset_stats"
    MARKET_GLOBAL = "market_global"
    NEWS_ARTICLES = "news_articles"
    SOCIAL_POSTS = "social_posts"
    INGESTION_SOURCES = "ingestion_sources"
    INGESTION_LOGS = "ingestion_logs"  # capped
    # module 3 - on-chain
    WHALE_TRANSFERS = "whale_transfers"
    WALLETS = "wallets"
    WALLET_BLACKLIST = "wallet_blacklist"
    ONCHAIN_METRICS = "onchain_metrics"  # time-series
    # module 4 - sentiment
    SENTIMENT_SNAPSHOTS = "sentiment_snapshots"  # time-series
    GEO_EVENTS = "geo_events"
    # module 5 - prediction
    PREDICTIONS = "predictions"
    VOLATILITY_FORECASTS = "volatility_forecasts"
    TRADE_SIGNALS = "trade_signals"
    ML_MODELS = "ml_models"
    # module 6 - fraud
    CONTRACT_SCANS = "contract_scans"
    FRAUD_ALERTS = "fraud_alerts"
    WATCHED_PAIRS = "watched_pairs"
    LIQUIDITY_SNAPSHOTS = "liquidity_snapshots"  # time-series
    PROJECT_SCORES = "project_scores"
    # module 7 - admin
    MODEL_CONTROLS = "model_controls"
    SYSTEM_METRICS = "system_metrics"  # time-series
    SCHEDULER_JOBS = "scheduler_jobs"
    WORKER_STATE = "worker_state"
    # module 8 - research
    REPORTS = "reports"
    CHAT_SESSIONS = "chat_sessions"
    CHAT_MESSAGES = "chat_messages"
    MARKET_INSIGHTS = "market_insights"
    # module 9 - trading
    EXCHANGE_CREDENTIALS = "exchange_credentials"
    ORDERS = "orders"
    FILLS = "fills"
    POSITIONS = "positions"
    PAPER_ACCOUNTS = "paper_accounts"
    PORTFOLIO_SNAPSHOTS = "portfolio_snapshots"  # time-series


@dataclass
class TimeSeriesSpec:
    name: str
    time_field: str = "ts"
    meta_field: str = "meta"
    granularity: str = "seconds"
    expire_after_seconds: int | None = None
    indexes: list[IndexModel] = field(default_factory=list)


def _idx(*keys: tuple[str, int], unique: bool = False, name: str | None = None, **kwargs: Any) -> IndexModel:
    return IndexModel(list(keys), unique=unique, name=name, **kwargs)


def _ttl(field_name: str, seconds: int) -> IndexModel:
    return IndexModel([(field_name, ASCENDING)], expireAfterSeconds=seconds)


REGULAR_INDEXES: dict[str, list[IndexModel]] = {
    Collections.USERS: [
        _idx(("email", ASCENDING), unique=True),
        _idx(("oauth_accounts.provider", ASCENDING), ("oauth_accounts.provider_user_id", ASCENDING), sparse=True),
        _idx(("role", ASCENDING)),
    ],
    Collections.REFRESH_TOKENS: [
        _idx(("jti", ASCENDING), unique=True),
        _idx(("user_id", ASCENDING), ("created_at", DESCENDING)),
        _idx(("family_id", ASCENDING)),
        _ttl("expires_at", 0),
    ],
    Collections.AUDIT_LOGS: [
        _idx(("seq", ASCENDING), unique=True),
        _idx(("actor_id", ASCENDING), ("created_at", DESCENDING)),
        _idx(("action", ASCENDING), ("created_at", DESCENDING)),
        _idx(("created_at", DESCENDING)),
    ],
    Collections.API_KEYS: [
        _idx(("key_hash", ASCENDING), unique=True),
        _idx(("user_id", ASCENDING)),
    ],
    Collections.DEVICE_TOKENS: [
        _idx(("token", ASCENDING), unique=True),
        _idx(("user_id", ASCENDING)),
    ],
    Collections.NOTIFICATIONS: [
        _idx(("user_id", ASCENDING), ("created_at", DESCENDING)),
        _idx(("user_id", ASCENDING), ("read_at", ASCENDING)),
        _ttl("created_at", 90 * 86400),
    ],
    Collections.ASSETS: [
        _idx(("symbol", ASCENDING), unique=True),
        _idx(("coingecko_id", ASCENDING), sparse=True),
    ],
    Collections.CANDLES: [
        _idx(("symbol", ASCENDING), ("interval", ASCENDING), ("open_time", DESCENDING), unique=True),
    ],
    Collections.ASSET_STATS: [_idx(("symbol", ASCENDING), unique=True), _idx(("rank", ASCENDING))],
    Collections.NEWS_ARTICLES: [
        _idx(("url_hash", ASCENDING), unique=True),
        _idx(("published_at", DESCENDING)),
        _idx(("analysis.status", ASCENDING), ("published_at", DESCENDING)),
        _idx(("symbols", ASCENDING), ("published_at", DESCENDING)),
        _idx(("title", "text"), ("summary", "text"), name="news_text"),
    ],
    Collections.SOCIAL_POSTS: [
        _idx(("external_id", ASCENDING), unique=True),
        _idx(("content_hash", ASCENDING)),
        _idx(("posted_at", DESCENDING)),
        _idx(("analysis.status", ASCENDING), ("posted_at", DESCENDING)),
        _idx(("symbols", ASCENDING), ("posted_at", DESCENDING)),
    ],
    Collections.INGESTION_SOURCES: [_idx(("name", ASCENDING), unique=True)],
    Collections.WHALE_TRANSFERS: [
        _idx(("tx_hash", ASCENDING), ("log_index", ASCENDING), unique=True),
        _idx(("block_time", DESCENDING)),
        _idx(("symbol", ASCENDING), ("block_time", DESCENDING)),
        _idx(("from_address", ASCENDING), ("block_time", DESCENDING)),
        _idx(("to_address", ASCENDING), ("block_time", DESCENDING)),
        _idx(("risk_level", ASCENDING), ("block_time", DESCENDING)),
    ],
    Collections.WALLETS: [
        _idx(("address", ASCENDING), unique=True),
        _idx(("smart_money_score", DESCENDING)),
        _idx(("cluster_id", ASCENDING), sparse=True),
    ],
    Collections.WALLET_BLACKLIST: [_idx(("address", ASCENDING), unique=True)],
    Collections.GEO_EVENTS: [
        _idx(("occurred_at", DESCENDING)),
        _idx(("region", ASCENDING), ("occurred_at", DESCENDING)),
        _idx(("fingerprint", ASCENDING), unique=True),
    ],
    Collections.PREDICTIONS: [
        _idx(("symbol", ASCENDING), ("horizon", ASCENDING), ("created_at", DESCENDING)),
        _idx(("created_at", DESCENDING)),
        _idx(("target_time", ASCENDING), ("evaluated", ASCENDING)),
    ],
    Collections.VOLATILITY_FORECASTS: [_idx(("symbol", ASCENDING), ("created_at", DESCENDING))],
    Collections.TRADE_SIGNALS: [
        _idx(("symbol", ASCENDING), ("created_at", DESCENDING)),
        _idx(("created_at", DESCENDING)),
    ],
    Collections.ML_MODELS: [
        _idx(("name", ASCENDING), ("symbol", ASCENDING), ("version", DESCENDING), unique=True),
    ],
    Collections.CONTRACT_SCANS: [
        _idx(("address", ASCENDING), ("created_at", DESCENDING)),
        _idx(("user_id", ASCENDING), ("created_at", DESCENDING)),
    ],
    Collections.FRAUD_ALERTS: [
        _idx(("created_at", DESCENDING)),
        _idx(("severity", ASCENDING), ("created_at", DESCENDING)),
        _idx(("token_address", ASCENDING), ("created_at", DESCENDING)),
        _idx(("fingerprint", ASCENDING), unique=True),
    ],
    Collections.WATCHED_PAIRS: [_idx(("chain", ASCENDING), ("pair_address", ASCENDING), unique=True)],
    Collections.PROJECT_SCORES: [_idx(("chain", ASCENDING), ("address", ASCENDING), unique=True)],
    Collections.MODEL_CONTROLS: [_idx(("name", ASCENDING), unique=True)],
    Collections.SCHEDULER_JOBS: [_idx(("name", ASCENDING), unique=True)],
    Collections.WORKER_STATE: [_idx(("key", ASCENDING), unique=True)],
    Collections.REPORTS: [
        _idx(("user_id", ASCENDING), ("created_at", DESCENDING)),
        _idx(("status", ASCENDING)),
    ],
    Collections.CHAT_SESSIONS: [_idx(("user_id", ASCENDING), ("updated_at", DESCENDING))],
    Collections.CHAT_MESSAGES: [_idx(("session_id", ASCENDING), ("created_at", ASCENDING))],
    Collections.MARKET_INSIGHTS: [_idx(("period", ASCENDING), ("period_start", DESCENDING), unique=True)],
    Collections.EXCHANGE_CREDENTIALS: [_idx(("user_id", ASCENDING), ("exchange", ASCENDING), unique=True)],
    Collections.ORDERS: [
        _idx(("user_id", ASCENDING), ("created_at", DESCENDING)),
        _idx(("user_id", ASCENDING), ("status", ASCENDING)),
        _idx(("status", ASCENDING), ("symbol", ASCENDING)),
        _idx(("client_order_id", ASCENDING), unique=True),
    ],
    Collections.FILLS: [_idx(("order_id", ASCENDING)), _idx(("user_id", ASCENDING), ("created_at", DESCENDING))],
    Collections.POSITIONS: [_idx(("user_id", ASCENDING), ("mode", ASCENDING), ("symbol", ASCENDING), unique=True)],
    Collections.PAPER_ACCOUNTS: [_idx(("user_id", ASCENDING), unique=True)],
}


def time_series_specs(settings: Settings) -> list[TimeSeriesSpec]:
    return [
        TimeSeriesSpec(
            Collections.PRICE_TICKS,
            granularity="seconds",
            expire_after_seconds=settings.mongodb_tick_ttl_days * 86400,
            indexes=[_idx(("meta.symbol", ASCENDING), ("ts", DESCENDING))],
        ),
        TimeSeriesSpec(
            Collections.SENTIMENT_SNAPSHOTS,
            granularity="minutes",
            indexes=[_idx(("meta.scope", ASCENDING), ("meta.symbol", ASCENDING), ("ts", DESCENDING))],
        ),
        TimeSeriesSpec(
            Collections.ONCHAIN_METRICS,
            granularity="minutes",
            indexes=[_idx(("meta.metric", ASCENDING), ("meta.symbol", ASCENDING), ("ts", DESCENDING))],
        ),
        TimeSeriesSpec(
            Collections.LIQUIDITY_SNAPSHOTS,
            granularity="minutes",
            expire_after_seconds=90 * 86400,
            indexes=[_idx(("meta.pair_address", ASCENDING), ("ts", DESCENDING))],
        ),
        TimeSeriesSpec(
            Collections.SYSTEM_METRICS,
            granularity="minutes",
            expire_after_seconds=14 * 86400,
        ),
        TimeSeriesSpec(
            Collections.PORTFOLIO_SNAPSHOTS,
            granularity="minutes",
            indexes=[_idx(("meta.user_id", ASCENDING), ("ts", DESCENDING))],
        ),
    ]


CAPPED_COLLECTIONS: dict[str, int] = {
    Collections.INGESTION_LOGS: 5 * 1024 * 1024,
}


async def ensure_schema(db: AsyncDatabase[dict[str, Any]], settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    existing = set(await db.list_collection_names())

    for spec in time_series_specs(settings):
        if spec.name not in existing:
            options: dict[str, Any] = {
                "timeseries": {
                    "timeField": spec.time_field,
                    "metaField": spec.meta_field,
                    "granularity": spec.granularity,
                }
            }
            if spec.expire_after_seconds:
                options["expireAfterSeconds"] = spec.expire_after_seconds
            try:
                await db.create_collection(spec.name, **options)
                logger.info("timeseries_collection_created", collection=spec.name)
            except CollectionInvalid:
                pass
            except OperationFailure as exc:  # pragma: no cover - old server without time-series
                logger.warning("timeseries_unsupported_fallback", collection=spec.name, error=str(exc))
                await db.create_collection(spec.name)
        if spec.indexes:
            await db[spec.name].create_indexes(spec.indexes)

    for name, size in CAPPED_COLLECTIONS.items():
        if name not in existing:
            try:
                await db.create_collection(name, capped=True, size=size)
            except CollectionInvalid:
                pass

    for name, indexes in REGULAR_INDEXES.items():
        try:
            await db[name].create_indexes(indexes)
        except OperationFailure as exc:
            # An index definition changed between versions; drop and recreate.
            logger.warning("index_conflict_recreating", collection=name, error=str(exc))
            await db[name].drop_indexes()
            await db[name].create_indexes(indexes)

    logger.info(
        "schema_ready", collections=len(REGULAR_INDEXES) + len(CAPPED_COLLECTIONS) + len(time_series_specs(settings))
    )


__all__ = ["REGULAR_INDEXES", "Collections", "TimeSeriesSpec", "ensure_schema", "time_series_specs"]
