"""Module 9 - Trading & Order Execution.

* ``paper.py``     - always-on paper trading engine (matching, cash/position checks, fills)
* ``exchange.py``  - CCXT adapter + live executor for real exchanges (encrypted user keys)
* ``portfolio.py`` - fill ledger, valuation / PnL, portfolio snapshots and history
* ``service.py``   - request-scoped business logic used by ``router.py``
* ``jobs.py``      - worker registration (paper matcher stream, sweep, snapshots, live sync)
"""
