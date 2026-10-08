# Developer shortcuts (GNU make). PowerShell equivalents are listed in README.md.
PYTHON ?= .venv/Scripts/python
ifeq ($(OS),Windows_NT)
PYTHON := .venv/Scripts/python
else
PYTHON := .venv/bin/python
endif

.PHONY: install dev api worker test lint format seed admin docker-up docker-down

install:        ## Create the virtualenv and install dev dependencies
	python -m venv .venv
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements-dev.txt

dev:            ## API with hot reload and background workers in-process
	RUN_WORKERS_IN_API=true $(PYTHON) -m uvicorn app.main:app --reload

api:            ## API only
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

worker:         ## Background worker only
	$(PYTHON) -m app.worker

test:           ## Run the test suite (embedded MongoDB unless MONGODB_TEST_URI is set)
	PYMONGOIM__MONGO_VERSION=7.0 $(PYTHON) -m pytest -q

lint:           ## Lint + format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:         ## Auto-fix lint issues and format
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .

seed:           ## Seed demo data (synthetic candles, news, admin user)
	$(PYTHON) -m scripts.seed_demo

admin:          ## Create/promote an admin: make admin EMAIL=you@example.com PASSWORD=...
	$(PYTHON) -m scripts.create_admin --email $(EMAIL) --password $(PASSWORD)

docker-up:      ## Start MongoDB + API + worker with Docker Compose
	docker compose up -d --build

docker-down:
	docker compose down
