# quantachain-backend

FastAPI backend for QuantaChain.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
```

- API root: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/api/v1/health
- Swagger docs: http://127.0.0.1:8000/docs

## Test & lint

```bash
pytest
ruff check .
ruff format .
```

## Project layout

```
app/
  main.py              # app factory + middleware
  core/config.py       # settings (loaded from .env)
  api/v1/router.py     # v1 router aggregation
  api/v1/endpoints/    # route modules
tests/                 # pytest suite
```

`.docs/` is a local-only folder for notes; it is git-ignored.
