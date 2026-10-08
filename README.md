# Quantachain Backend

AI-powered crypto intelligence platform — FastAPI + MongoDB backend for the Quantachain
final-year project (COMSATS University Islamabad). It unifies live market data, on-chain whale
tracking, GPT-based sentiment and geopolitical analysis, price forecasting, fraud detection and
trading behind one API, with an admin control plane on top.

## Modules

| # | Module | Status | Highlights |
|---|--------|--------|------------|
| 1 | Authentication & Access Control | ✅ | Email/password + Google/GitHub OAuth 2.0, JWT access/refresh rotation with reuse detection, TOTP MFA (Google Authenticator) with recovery codes, role-based access (admin / user), API keys, tamper-evident hash-chained audit trail |
| 2 | Multi-Source Data Aggregation | ✅ | Binance WebSocket ticks → MongoDB time-series, 1m candles rolled up to 5m/15m/1h/4h/1d, REST back-fill, CoinGecko + Fear & Greed overview, NewsAPI/RSS/Reddit ingestion with canonical-URL and content dedup, Data Ingestion Hub status |
| 3 | On-Chain Intelligence Engine | 🚧 | Whale transfer detection (ETH + ERC-20), exchange flow labelling, smart-money scoring, wallet cluster graphs, Alchemy → Infura → public RPC failover, Moralis streams |
| 4 | Sentiment & Geopolitical Analysis | 🚧 | GPT-4o scoring + NER with heuristic fallback, sentiment snapshots ("Vibe Index"), sentiment/price correlation, geopolitical impact heatmap |
| 5 | AI Prediction & Forecasting | 🚧 | LSTM (TensorFlow, optional) / statistical ensemble, volatility regimes, confidence-scored PDF reports, automated trade signals |
| 6 | Risk Assessment & Fraud Detection | 🚧 | Solidity vulnerability scanner (SWC rules), rug-pull / pump-and-dump detection from DEX liquidity, Transparency & Liquidity Score, FCM alerts, AML blacklist |
| 7 | Admin Dashboard & Monitoring | ✅ | CPU/RAM/event-loop metrics, health dashboard, job scheduler controls, AI model pause/override, user management, audit export, error e-mail alerts |
| 8 | Interactive Research Portal | ✅ | Aligned multi-metric chart series, multi-turn AI research assistant with tool calling (works without an API key), monthly/quarterly macro insights |
| 9 | Trading & Order Execution | 🚧 | Paper trading engine (market/limit/stop-loss), CCXT live adapter with encrypted exchange keys, real-time portfolio PnL |

Cross-cutting: in-app + FCM push notifications, WebSocket streams (`/ws/market`, `/ws/onchain`,
`/ws/alerts`, `/ws/portfolio`, `/ws/ingestion`), persisted job scheduler, structured logging,
per-provider usage metrics. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and
module contract.

## Quick start

Requirements: Python 3.11+ and a MongoDB 7 (Docker, local install, or a free Atlas cluster).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                 # cp on macOS/Linux; set MONGODB_URI and SECRET_KEY
```

Start MongoDB (pick one):

```bash
docker compose up -d mongo             # local container on :27017
# or set MONGODB_URI to an Atlas connection string in .env
```

Run the API with the background workers in-process (simplest for development):

```bash
RUN_WORKERS_IN_API=true uvicorn app.main:app --reload
# PowerShell:  $env:RUN_WORKERS_IN_API="true"; uvicorn app.main:app --reload
```

Or run them separately (recommended for anything beyond local development):

```bash
uvicorn app.main:app --reload          # API
python -m app.worker                   # ingestion, AI batches, forecasts, scans
```

- Swagger UI: http://127.0.0.1:8000/docs · ReDoc: http://127.0.0.1:8000/redoc
- Health: http://127.0.0.1:8000/api/v1/health

Seed demo data (synthetic candles, sample news, an admin account) so every screen has content
without any API keys:

```bash
python -m scripts.seed_demo            # admin@quantachain.local / Admin1234!
python -m scripts.create_admin --email you@example.com --password 'Str0ngPass!'
```

In development the first registered user is automatically promoted to admin
(`FIRST_ADMIN_EMAIL` does the same in production).

## Configuration

Everything is read from environment variables / `.env` — see [.env.example](.env.example) for the
full, commented list. Every third-party key is optional; features degrade gracefully:

| Provider | Used for | Without it |
|----------|----------|------------|
| Binance / CoinGecko (public) | prices, candles, market overview | — (no key needed) |
| `NEWSAPI_KEY` | global headlines | RSS feeds only |
| `OPENAI_API_KEY` | GPT-4o sentiment, NER, research assistant, narratives | heuristic analyzer + rule-based assistant |
| `ALCHEMY_API_KEY` / `INFURA_API_KEY` | primary EVM RPC with failover | public RPC endpoints |
| `MORALIS_API_KEY` | wallet/token enrichment, streams | block scanning only |
| `ETHERSCAN_API_KEY` | verified contract source | bytecode-only scanning |
| `FCM_SERVICE_ACCOUNT_JSON` | mobile push | in-app + WebSocket alerts |
| `SMTP_*` | admin e-mail alerts | in-app admin notifications |
| TensorFlow (`requirements-ml.txt`) | LSTM forecaster | statistical ensemble |

Live trading is off by default (`LIVE_TRADING_ENABLED=false`); paper trading is always available.

## API overview

All routes are under `/api/v1` and documented in Swagger. Authenticate with
`Authorization: Bearer <access token>` or `X-API-Key: qc_...`.

| Area | Endpoints (abridged) |
|------|----------------------|
| Auth | `POST /auth/register`, `/auth/login`, `/auth/mfa/verify`, `/auth/refresh`, `/auth/logout`, `GET/PATCH /auth/me`, `/auth/mfa/*`, `/auth/oauth/{google,github}/start`, `/auth/api-keys`, `/auth/sessions`, `/auth/audit` |
| Market | `GET /market/overview`, `/market/assets?tab=hot|gainers|losers|volume|new`, `/market/assets/{symbol}`, `/market/assets/{symbol}/candles?interval=1h`, `/market/news`, `/market/social`, admin `/market/ingestion` |
| Research | `GET /research/charts/{symbol}?metrics=price,sentiment,netflow`, `POST /research/chat/sessions/{id}/messages`, `/research/chat/quick`, `/research/library`, `/research/insights` |
| Admin | `GET /admin/overview`, `/admin/health`, `/admin/system/metrics`, `/admin/jobs`, `/admin/models`, `/admin/users`, `/admin/audit-logs(/export)`, `/admin/logs/export`, `POST /admin/alerts/test` |
| Notifications | `GET /notifications`, `POST /notifications/devices` (FCM token) |

Errors always have the shape `{"error": {"code", "message", "details", "request_id"}}`.

## Development

```bash
ruff check . && ruff format --check .   # lint
pytest -q                                # tests (embedded MongoDB downloaded on first run,
                                         # or set MONGODB_TEST_URI to reuse a running server)
make dev | make test | make lint         # shortcuts (GNU make)
```

Tests run against a real MongoDB 7 (time-series collections included) and never touch the
network; provider clients are faked. CI (GitHub Actions) runs lint, the test matrix on Python
3.11/3.12 with a MongoDB service, and a Docker image build.

## Deployment

```bash
docker compose up -d --build           # mongo + api + worker
docker compose --profile tools up -d   # adds mongo-express on :8081
```

The image (`Dockerfile`) runs as a non-root user with a health check; the worker uses the same
image with `python -m app.worker`. Generated files (ML models, PDF reports) live in `DATA_DIR`
(`/app/var` in the container).

## Project layout

```
app/
  main.py              FastAPI factory + lifespan      app/worker.py   background worker entrypoint
  core/                settings, logging, security, events, metrics, rate limiting
  db/                  Mongo client, Document/Repository base, collections & indexes
  api/                 dependencies (auth/RBAC), router aggregation, health, WebSockets
  integrations/        Binance, CoinGecko, NewsAPI/RSS/Reddit, EVM RPC, Moralis, Etherscan,
                       DexScreener, OpenAI, FCM, SMTP
  workers/             scheduler, ingestion source tracker, runtime
  modules/<module>/    models · repository · schemas · service · router · jobs · tests
scripts/               seed_demo, create_admin
docs/ARCHITECTURE.md   design, module contract, data contracts
```

`.docs/` holds private project documents (the scope document, notes) and is git-ignored.
