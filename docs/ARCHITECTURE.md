# Quantachain Backend - Architecture

FastAPI + MongoDB backend for the Quantachain AI crypto-intelligence platform
(COMSATS FYP, FA23-BSE). Two processes share one codebase:

| Process | Command | Role |
|---|---|---|
| API | `uvicorn app.main:app` | REST + WebSocket endpoints, auth, read models |
| Worker | `python -m app.worker` | Ingestion streams, polling jobs, AI batches, forecasts |

Set `RUN_WORKERS_IN_API=true` to run both in one process during development.

```
app/
  main.py                 FastAPI factory, lifespan (Mongo connect, schema, WS hub)
  worker.py               Standalone worker entrypoint
  core/                   config, logging, errors, security, event bus, metrics, pagination
  db/                     Mongo client, Document base model, generic Repository, schema/indexes
  api/                    deps (auth/RBAC), router aggregation, health, websockets
  integrations/           Thin clients for third-party APIs (httpx based, metered)
  workers/                Scheduler (persisted job state), source tracker, runtime
  modules/<module>/       One package per scope module (see below)
```

## Module layout (contract)

Every module under `app/modules/<name>/` follows the same shape:

| File | Purpose |
|---|---|
| `models.py` | Pydantic `Document` subclasses persisted in Mongo. `StrEnum` for enums, primitives only (no `HttpUrl`/`Decimal`). |
| `repository.py` | `Repository[Model]` subclasses (`collection_name`, `model`). Query helpers live here, never in routers. |
| `schemas.py` | API request/response models. Never expose internal fields (hashes, encrypted secrets). |
| `service.py` | Business logic. Constructed per request as `Service(db, settings)`; raises `app.core.exceptions.*`. |
| `router.py` | `router = APIRouter(prefix="/<name>", tags=["<name>"])`. Auto-included by `app/api/router.py`. |
| `jobs.py` | Optional `register(runtime: WorkerRuntime)` adding `JobSpec`s / streams. Auto-discovered by the runtime. |

Rules:

* Collection names and indexes are declared centrally in `app/db/collections.py` (`Collections.*`). Add new collections there.
* Settings are declared centrally in `app/core/config.py`. Read them via `get_settings()` / the `SettingsDep` dependency.
* Dependencies: `DB`, `CurrentUser`, `AdminUser`, `OptionalUser`, `Pagination` from `app.api.deps` / `app.core.pagination`.
* Lists are paginated with `Page[T].build(items, total, params)` (limit/offset).
* External providers go through `app/integrations/*` (`ProviderClient` records latency/errors for the admin dashboard). Every provider must be optional: when a key is missing the feature degrades gracefully (heuristic fallback or `FeatureDisabledError`), never crashes start-up.
* Events: publish on `app.core.events.event_bus` with `Topics.*`; WebSocket endpoints in `app/api/ws.py` fan them out. Payloads must be JSON-serialisable dicts; include `user_id` for user-scoped events and `symbol` for market events.
* User-facing alerts go through `NotificationService.notify(...)` / `notify_subscribers(...)` (`app/modules/notifications/service.py`), which stores the notification, publishes `Topics.ALERT` and pushes over FCM.
* Background work: `JobSpec(name, interval_seconds, func, module=..., description=..., run_on_start=..., timeout_seconds=...)`. Long-lived streams: `runtime.add_stream(name, coroutine_factory)`. Ingestion sources register with `runtime.tracker.register(...)` and report `event()/error()/set_status()` so the Data Ingestion Hub reflects them.
* Model pause/override (Module 7): long-running AI jobs must check `ModelControlService.is_paused("<model name>")` before running.
* Logging: `structlog.get_logger(__name__)`, snake_case event names, key/value context. No print.
* Tests: `tests/test_<module>.py` using the fixtures in `tests/conftest.py` (`client`, `db`, `user_account`, `admin_account`). Pure logic (scoring, detectors, parsers) gets unit tests without the database.

## Cross-module data contracts

**Sentiment snapshots** (`Collections.SENTIMENT_SNAPSHOTS`, time-series, written by Module 4, read by Modules 2/5/8):

```json
{"ts": <datetime>, "meta": {"scope": "global" | "asset", "symbol": "BTC" | null},
 "score": -1..1, "index": 0..100, "label": "very_bearish|bearish|neutral|bullish|very_bullish",
 "sample_size": <int>, "breakdown": {"retail": <score>, "institutional": <score>, "whale": <score>}}
```

**Text analysis** (`TextAnalysis` in `app/modules/market/models.py`) is embedded in `news_articles.analysis` and
`social_posts.analysis`; Module 4 fills it and flips `status` from `pending` to `done`.

**Candles** (`Collections.CANDLES`): `{symbol, interval (1m|5m|15m|1h|4h|1d), open_time, close_time, open, high, low, close, volume, quote_volume, trades, closed}`.
Use `CandleRepository.range()` / `.closes()` for features. Latest price: `MarketService.latest_price(symbol)`.

**Whale transfers** (`Collections.WHALE_TRANSFERS`, Module 3): one document per transfer with
`symbol, amount, amount_usd, from_address, to_address, from_label, to_label, flow (exchange_inflow|exchange_outflow|exchange_internal|wallet_to_wallet), risk_level (low|medium|suspicious), block_time`.

**On-chain metrics** (`Collections.ONCHAIN_METRICS`, time-series): `{ts, meta: {metric, symbol}, value}` -
metrics include `exchange_netflow_usd`, `active_whales`, `observed_active_addresses`.

**Predictions** (`Collections.PREDICTIONS`, Module 5): `{symbol, horizon (1h|4h|24h), target_price, lower, upper, confidence, direction, model, created_at, target_time}`.

**Fraud alerts** (`Collections.FRAUD_ALERTS`, Module 6): `{type (rug_pull|pump_and_dump|contract_vulnerability|blacklisted_wallet), severity (info|warning|critical), token_address, pair_address, title, body, evidence, fingerprint}`.

## Request lifecycle

1. `RequestContextMiddleware` assigns `X-Request-ID`, times the request, records metrics and logs an access line.
2. Auth: `Authorization: Bearer <access JWT>` or `X-API-Key: qc_...`. Access tokens carry `role` and `ver` (token version) so "log out everywhere" works instantly.
3. Errors are rendered uniformly: `{"error": {"code", "message", "details", "request_id"}}`.

## Data flow

```
Binance WS ─┐                      ┌─> /ws/market, Markets table, Trade terminal
CoinGecko  ─┼─> Module 2 (Mongo) ──┼─> Module 4 sentiment ──> snapshots ──> Module 5 features ──> predictions/signals
NewsAPI/RSS─┘                      └─> Module 8 charts/reports/chat
Alchemy/Infura/Moralis ──> Module 3 whale feed ──> alerts (FCM / WS) ──> Module 6 risk scoring
DexScreener/Etherscan  ──> Module 6 scanner & liquidity monitor ──> fraud alerts
```
