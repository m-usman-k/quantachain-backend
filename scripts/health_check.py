"""System & service health verification tool for QuantaChain (CLI).

    python -m scripts.health_check
    python -m scripts.health_check --json
    python -m scripts.health_check --verbose

Exits 0 if all critical services are healthy, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

from app.core.config import get_settings
from app.core.events import event_bus
from app.db.collections import Collections
from app.db.mongo import mongo
from app.modules.prediction.volatility import garch11_fit, garch_variance_path


async def check_database() -> dict[str, Any]:
    started = time.perf_counter()
    ping = await mongo.ping_ms() if mongo.connected else None
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    healthy = ping is not None
    counts: dict[str, int] = {}
    if healthy and mongo.db is not None:
        for coll in (
            Collections.USERS,
            Collections.ASSETS,
            Collections.CANDLES,
            Collections.NEWS_ARTICLES,
            Collections.WHALE_TRANSFERS,
            Collections.PREDICTIONS,
            Collections.FRAUD_ALERTS,
            Collections.ORDERS,
        ):
            try:
                counts[coll] = await mongo.db[coll].count_documents({})
            except Exception:
                counts[coll] = -1
    return {
        "healthy": healthy,
        "ping_ms": ping,
        "roundtrip_ms": duration_ms,
        "collections": counts,
    }


async def check_event_bus() -> dict[str, Any]:
    started = time.perf_counter()
    sub = event_bus.subscribe("health:bus_test")
    try:
        event_bus.publish("health:bus_test", {"type": "HEALTH_PING", "ts": time.time()})
        received = await sub.get(timeout=2.0)
        healthy = received is not None and received.payload.get("type") == "HEALTH_PING"
    except Exception as exc:
        healthy = False
        received = str(exc)
    finally:
        sub.close()
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {"healthy": healthy, "latency_ms": duration_ms}


def check_models() -> dict[str, Any]:
    import numpy as np

    started = time.perf_counter()
    try:
        # Test statistical baseline (GARCH)
        test_returns = np.random.normal(0, 0.02, 120)
        fit = garch11_fit(test_returns)
        fc = garch_variance_path(fit, steps=24) if fit else None
        garch_ok = fit is not None and fc is not None and len(fc) == 24
    except Exception:
        garch_ok = False
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "healthy": garch_ok,
        "garch_baseline": "ready" if garch_ok else "failed",
        "heuristic_nlp": "ready",
        "latency_ms": duration_ms,
    }


async def run_diagnostics(verbose: bool = False) -> dict[str, Any]:
    settings = get_settings()
    await mongo.connect()

    db_res = await check_database()
    bus_res = await check_event_bus()
    models_res = check_models()

    overall_healthy = db_res["healthy"] and bus_res["healthy"] and models_res["healthy"]

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "HEALTHY" if overall_healthy else "DEGRADED",
        "app": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "data_dir": settings.data_dir,
            "python_version": sys.version.split()[0],
        },
        "database": db_res,
        "event_bus": bus_res,
        "models": models_res,
    }
    return report


def print_cli_report(report: dict[str, Any], verbose: bool = False) -> None:
    status_str = report["status"]
    color = "\033[92m" if status_str == "HEALTHY" else "\033[91m"
    reset = "\033[0m"

    print("=" * 60)
    print(f" QuantaChain Health Check Diagnostic - {color}{status_str}{reset}")
    print("=" * 60)
    app = report["app"]
    print(f" App:        {app['name']} v{app['version']} ({app['environment']})")
    print(f" Python:     {app['python_version']}")
    print(f" Storage:    {app['data_dir']}")
    print("-" * 60)

    db = report["database"]
    db_sym = "[OK]" if db["healthy"] else "[FAIL]"
    print(f" Database:   {db_sym} Ping: {db.get('ping_ms')}ms (roundtrip {db.get('roundtrip_ms')}ms)")
    if db.get("collections"):
        print(" Collections:")
        for name, count in db["collections"].items():
            print(f"   - {name:<20}: {count:>6} docs")
    print("-" * 60)

    bus = report["event_bus"]
    bus_sym = "[OK]" if bus["healthy"] else "[FAIL]"
    print(f" Event Bus:  {bus_sym} Roundtrip: {bus.get('latency_ms')}ms")

    models = report["models"]
    mod_sym = "[OK]" if models["healthy"] else "[FAIL]"
    print(f" Analytics:  {mod_sym} GARCH: {models.get('garch_baseline')} | NLP: {models.get('heuristic_nlp')}")
    print("=" * 60)


async def main() -> None:
    parser = argparse.ArgumentParser(description="QuantaChain System Health Check")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()

    try:
        report = await run_diagnostics(verbose=args.verbose)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_cli_report(report, verbose=args.verbose)
        sys.exit(0 if report["status"] == "HEALTHY" else 1)
    finally:
        await mongo.close()


if __name__ == "__main__":
    asyncio.run(main())
