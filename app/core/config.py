"""Application settings.

Every tunable lives here and is read from the environment (or a local ``.env``
file). Grouped by concern so the file doubles as documentation of what the
platform can be configured to do. Secrets use ``SecretStr`` so they never leak
into logs or ``repr`` output.
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# A list field that accepts either JSON (`["a","b"]`) or a comma separated
# string (`a,b`) from the environment.
CsvList = Annotated[list[str], NoDecode]

Environment = Literal["development", "test", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Quantachain Backend"
    app_version: str = "0.1.0"
    environment: Environment = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool | None = None  # None -> JSON in production, pretty console elsewhere
    api_v1_prefix: str = "/api/v1"
    cors_origins: CsvList = ["http://localhost:3000"]
    public_base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    docs_enabled: bool = True

    # ------------------------------------------------------------- database
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "quantachain"
    mongodb_timeout_ms: int = 5000
    mongodb_tick_ttl_days: int = 30  # raw price ticks are expired by MongoDB after this

    # ------------------------------------------------------------- security
    secret_key: SecretStr = SecretStr("change-me-this-is-not-a-secret")
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 14
    mfa_token_ttl_minutes: int = 5
    mfa_issuer: str = "Quantachain"
    encryption_key: SecretStr | None = None  # Fernet key; derived from secret_key when unset
    password_min_length: int = 8
    rate_limit_per_minute: int = 240  # per user / IP; 0 disables
    first_admin_email: str | None = None  # this email becomes admin on registration

    # ---------------------------------------------------------------- oauth
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    github_client_id: str | None = None
    github_client_secret: SecretStr | None = None

    # -------------------------------------------------------------- workers
    workers_enabled: bool = True  # the dedicated worker process runs jobs
    run_workers_in_api: bool = False  # dev convenience: run ingestion inside the API
    backfill_on_start: bool = True  # pull historical candles when the worker starts
    tracked_symbols: CsvList = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT"]
    quote_asset: str = "USDT"

    # ------------------------------------------------------ market data (M2)
    binance_ws_enabled: bool = True
    binance_ws_url: str = "wss://stream.binance.com:9443/stream"
    binance_rest_url: str = "https://api.binance.com"
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"
    coingecko_api_key: SecretStr | None = None
    coingecko_poll_seconds: int = 120
    fear_greed_api_url: str = "https://api.alternative.me/fng/"
    newsapi_key: SecretStr | None = None
    newsapi_url: str = "https://newsapi.org/v2"
    news_poll_seconds: int = 300
    news_query: str = "crypto OR bitcoin OR ethereum OR blockchain OR stablecoin OR SEC crypto"
    rss_feeds: CsvList = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/rss.xml",
    ]
    reddit_enabled: bool = True
    reddit_subreddits: CsvList = ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets"]
    social_poll_seconds: int = 300

    # --------------------------------------------------------- on-chain (M3)
    alchemy_api_key: SecretStr | None = None
    infura_api_key: SecretStr | None = None
    moralis_api_key: SecretStr | None = None
    moralis_stream_secret: SecretStr | None = None
    etherscan_api_key: SecretStr | None = None
    etherscan_api_url: str = "https://api.etherscan.io/v2/api"
    evm_chain_id: int = 1
    evm_rpc_urls: CsvList = []  # extra JSON-RPC fallbacks, tried after Alchemy/Infura
    evm_public_rpc_urls: CsvList = ["https://ethereum-rpc.publicnode.com", "https://eth.llamarpc.com"]
    onchain_poll_seconds: int = 15
    onchain_max_blocks_per_poll: int = 5
    whale_threshold_usd: float = 1_000_000.0
    whale_alert_threshold_usd: float = 10_000_000.0

    # --------------------------------------------------------- sentiment (M4)
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o"
    openai_base_url: str | None = None
    llm_timeout_seconds: float = 60.0
    sentiment_batch_size: int = 20
    sentiment_poll_seconds: int = 120
    sentiment_snapshot_seconds: int = 600

    # -------------------------------------------------------- prediction (M5)
    data_dir: str = "var"  # models, reports and other generated files live here
    prediction_refresh_seconds: int = 900
    prediction_horizons: CsvList = ["1h", "4h", "24h"]
    lstm_enabled: bool = True  # used when TensorFlow is installed and a model is trained
    lstm_lookback: int = 48
    signal_sigma_threshold: float = 1.5

    # ------------------------------------------------------------ fraud (M6)
    dexscreener_api_url: str = "https://api.dexscreener.com"
    fraud_poll_seconds: int = 120
    rug_pull_liquidity_drop_pct: float = 50.0
    pump_price_spike_pct: float = 40.0
    pump_volume_multiplier: float = 4.0

    # ---------------------------------------------------------- notifications
    fcm_project_id: str | None = None
    fcm_service_account_json: SecretStr | None = None  # JSON string or path to the file
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    smtp_from: str = "alerts@quantachain.local"
    smtp_use_tls: bool = True
    admin_alert_emails: CsvList = []
    error_alert_threshold: int = 5  # 5xx responses within the window trigger an email
    error_alert_window_seconds: int = 300
    error_alert_cooldown_seconds: int = 1800

    # ---------------------------------------------------------- trading (M9)
    live_trading_enabled: bool = False  # global kill switch; paper trading is always on
    paper_starting_balance: float = 100_000.0
    paper_fee_rate: float = 0.001
    default_exchange: str = "binance"

    # ------------------------------------------------------------ validators
    @field_validator(
        "cors_origins",
        "tracked_symbols",
        "rss_feeds",
        "reddit_subreddits",
        "evm_rpc_urls",
        "evm_public_rpc_urls",
        "prediction_horizons",
        "admin_alert_emails",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                return json.loads(text)
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @field_validator("tracked_symbols", mode="after")
    @classmethod
    def _upper_symbols(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for symbol in value:
            upper = symbol.upper()
            if upper not in seen:
                seen.append(upper)
        return seen

    # ------------------------------------------------------------ helpers
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def log_as_json(self) -> bool:
        return self.is_production if self.log_json is None else self.log_json

    @property
    def fernet_key(self) -> bytes:
        """Key for encrypting user secrets (exchange API keys, MFA seeds)."""
        if self.encryption_key is not None:
            return self.encryption_key.get_secret_value().encode()
        digest = hashlib.sha256(self.secret_key.get_secret_value().encode()).digest()
        return base64.urlsafe_b64encode(digest)

    @property
    def openai_configured(self) -> bool:
        return self.openai_api_key is not None and bool(self.openai_api_key.get_secret_value())

    @property
    def symbols_with_quote(self) -> list[str]:
        return [f"{symbol}{self.quote_asset}" for symbol in self.tracked_symbols]

    def oauth_redirect_uri(self, provider: str) -> str:
        return f"{self.public_base_url.rstrip('/')}{self.api_v1_prefix}/auth/oauth/{provider}/callback"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that need to re-read the environment."""
    get_settings.cache_clear()


__all__ = ["Field", "Settings", "get_settings", "reset_settings_cache"]
