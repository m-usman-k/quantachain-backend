## Summary

<!-- What does this change and why? Link the module (M1-M9) and feature (FE-x) from the scope. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest -q` passes (new behaviour has tests)
- [ ] New settings documented in `.env.example`; new collections/indexes in `app/db/collections.py`
- [ ] Works without optional API keys (graceful degradation)
- [ ] No secrets, `.env` files or `.docs/` content committed
