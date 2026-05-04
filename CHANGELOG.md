# Changelog

All notable changes to Athena Agent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Agent loop with tool registry (Week 3)
- Skill system + background review fork (Week 3)
- Telegram messaging gateway (Week 3)
- Sandboxed Python REPL tool (Week 3)
- Live signal monitoring with cron (Week 4)
- arXiv q-fin knowledge ingestion (Week 4)

## [0.1.0] - 2026-05-05

Initial release. Data spine + research stack + LLM client. 35/35 tests passing,
no network required for any test.

### Added — Week 1: Data Spine

- Canonical data contract (`athena/tools/data/contract.py`) with strict
  validation: tz-aware UTC timestamps, monotonic, no duplicates, no NaNs in
  OHLC, high ≥ low.
- Canonical symbol scheme (`athena/tools/data/symbols.py`):
  `VENUE:ROOT[:YYYYMMDD[:STRIKE]:R]` covering equities, futures, and options
  uniformly.
- Pluggable local cache (`athena/tools/data/cache.py`) with parquet, feather,
  and pickle backends. Selectable via `ATHENA_CACHE_FORMAT` env var.
- Three data adapters speaking the canonical contract:
  - `CCXTAdapter` — Binance public market data (no auth required).
  - `DeltaAdapter` — Delta Exchange spot, perpetuals, dated futures, and
    options chains via public REST.
  - `DhanAdapter` — Dhan HQ v2 API for NSE/BSE equities, F&O, MCX commodities.
    Properly handles `EQUITY`, `INDEX`, `FUTSTK`, `FUTIDX`, `FUTCOM`, `OPTSTK`,
    `OPTIDX` instrument types.
- `DataRouter` — single entry point for the agent. Routes by venue prefix,
  manages cache reads, computes gaps, and serves overlapping requests from
  cache when possible.

### Added — Week 2: Research Stack

- Immutable run records (`athena/tools/research/runs.py`):
  - Content-addressed `run_id` scheme: `YYYYMMDD-HHMMSS-strategy-paramshash8`.
  - Atomic writes via staging directory.
  - `RunStore.write()` refuses to overwrite an existing run.
- Performance statistics (`athena/tools/research/stats.py`):
  - Sharpe, Sortino, Calmar, CAGR, max drawdown, hit rate, turnover.
  - **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) to correct for
    multiple-testing bias when sweeping parameters.
  - Probabilistic Sharpe Ratio (PSR) as the single-trial degenerate case.
  - Beasley-Springer-Moro inverse normal so SciPy is optional.
  - Crypto-aware annualization (365 days vs 252 trading days).
- vectorbt wrapper (`athena/tools/research/backtest_vbt.py`) for fast
  parameter sweeps. Takes the data router's output, returns a complete run
  record with stats and equity curve.
- backtrader wrapper (`athena/tools/research/backtest_bt.py`) for production
  validation with realistic fills. Per-interval `TimeFrame` mapping so
  intraday equity curves have correct granularity.
- SQLite registry (`athena/agent/store.py`):
  - `backtest_runs` — every run indexed for SQL query.
  - `signal_evaluations` — every signal evaluation logged with context.
  - `memories` — episodic + semantic memory the agent writes.
  - `skill_meta` — usage counters, pinning, archiving.
- OpenRouter LLM client (`athena/agent/llm.py`):
  - Three-tier model routing: `smart` (Sonnet), `cheap` (Haiku),
    `embeddings`.
  - Configurable via `ATHENA_MODEL_*` environment variables.
  - Retry with exponential backoff on 429s and 5xx errors.

### Tests

- `test_data_spine.py` — 7 tests covering symbol parsing, validation, cache
  round-trip, router cache-hit and gap-extension behavior, venue dispatch.
- `test_stats.py` — 17 tests pinning the math, especially the deflated Sharpe
  on engineered cases (strong-signal-low-trials passes, weak-signal-high-trials
  fails).
- `test_runs_and_store.py` — 10 tests covering immutable runs, atomic writes,
  staging recovery, and every SQLite store method.
- `test_e2e_pipeline.py` — 1 end-to-end test wiring data router → signal → stats
  → RunStore → SQLite. No network, no LLM, no backtest engines.
