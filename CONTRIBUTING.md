# Contributing to Athena Agent

Thanks for your interest in contributing. Athena is built primarily for
solo quants — the bar for adding code is "would I want this running in my
own research stack?"

## Getting set up

```bash
git clone https://github.com/<your-username>/athena-agent.git
cd athena-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,all]"
```

Run the test suite:

```bash
ATHENA_CACHE_FORMAT=pickle pytest athena/tests/ -v
```

All tests are network-free. If a test you write requires the network,
mark it with `@pytest.mark.network` and gate it behind an env var.

## Code style

- Python 3.10+ idioms (`match`, `|` unions, `from __future__ import annotations`).
- Run `ruff check athena/` and `ruff format athena/` before submitting.
- Type hints everywhere except in test files.
- Docstrings on public functions explaining *why*, not *what* (the code
  shows what it does).

## Design principles

These are non-negotiable. Patches that violate them get rejected even if
the code is good.

1. **The data contract is sacred.** Every adapter returns frames in the
   exact shape defined in `tools/data/contract.py`. No vendor-specific
   columns leak past the adapter boundary. If a vendor offers richer data,
   it goes in `meta`, not the frame.

2. **Runs are immutable.** Once a backtest is written to disk, it is never
   modified. Re-running the same params produces the same `run_id`. This
   is what makes "have I tested this before" answerable.

3. **Multiple-testing correction is mandatory.** Anything that reports
   Sharpe must also report deflated Sharpe with the appropriate `n_trials`.
   Naive Sharpe alone is a false-positive factory.

4. **The agent never trades autonomously.** It alerts. The human stays in
   the loop. Add execution only behind explicit, deliberate gates.

5. **Tests are network-free by default.** Use `FakeAdapter` and synthetic
   data. Real-venue tests are valuable but go in a separate
   `tests/integration/` directory and are opt-in.

## Adding a new data adapter

1. Implement the `DataAdapter` protocol from `contract.py` — four methods:
   `search`, `history`, `quote`, `chain`.
2. Convert vendor symbols to and from the canonical Athena scheme.
3. Call `validate_bars()` before returning any history frame.
4. Raise `NotSupported` for methods that don't apply (don't fake them).
5. Register in `router.build_default_router()` behind an `enable_<venue>`
   flag.
6. Add a test with a fake HTTP layer (mock `requests`), not a live call.

## Adding a new statistic

1. Implement in `tools/research/stats.py` as a pure function over a
   returns Series.
2. Add it to `compute_stats()` so every backtest captures it.
3. Add a test on a constructed series with a known answer.

## Adding a new tool to the agent (Week 3+)

Once the agent loop lands:

1. Add the tool function in the appropriate `tools/<domain>/` module.
2. Register via `tools.registry.register(...)` with a JSON-schema-style
   parameter spec.
3. Document the tool in `skills/` if it has non-obvious usage patterns.

## Pull request checklist

- [ ] Tests added or updated; all tests pass
- [ ] `ruff check` clean
- [ ] No secrets, broker tokens, or `.env` files in the diff
- [ ] CHANGELOG updated under `[Unreleased]`
- [ ] If changing the data contract, the change is backward-compatible or
  explicitly noted as breaking
- [ ] If touching the deflated Sharpe math, a test pins the new behavior
  on engineered cases

## Reporting bugs

Open a GitHub issue with:
- The minimal code that reproduces the bug
- What you expected vs what happened
- Python version, OS, and relevant package versions (`pip list | grep -E "pandas|vectorbt|backtrader"`)

For Dhan or Delta API quirks specifically: include the venue, symbol,
date range, and (redacted) error response. The vendor APIs evolve, so
"this worked last month" bugs are real.
