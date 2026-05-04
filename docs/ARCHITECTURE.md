# Architecture

This document explains *why* Athena is structured the way it is. The README
covers the *what*; this is the *why*.

## The keystone: a single data contract

Every data adapter — Dhan, Delta Exchange, CCXT, and any future venue —
returns DataFrames with this exact shape:

```
columns: [ts_utc, symbol, asset_class, venue,
          open, high, low, close, volume, oi]
ts_utc:  tz-aware UTC, monotonic, no duplicates, no NaN OHLC
```

Symbols are canonical:

```
NSE:RELIANCE                       Indian equity
NSE:NIFTY:20260529:24500:C         NIFTY call option
NSE:NIFTY:20260529:F               NIFTY future
DELTA:BTCUSD                       Delta perpetual
DELTA:BTCUSD:20260627:80000:C      Delta dated option
BINANCE:BTC/USDT                   CCXT spot
```

This sounds boring. It is the most important decision in the codebase.
Once strategies and signals are written against the contract, a pairs-trade
backtest works on NIFTY-BANKNIFTY, BTC-ETH, and EURUSD-GBPUSD without
rewriting. The agent's accumulated skills become portable across asset
classes for free. Vendor-specific quirks stay quarantined inside the
adapter.

## Why a router with a cache, not direct adapter calls

The naive design is "agent calls adapter." That breaks the moment you
start monitoring 50 symbols every 15 minutes — you'll burn through Dhan's
rate limits and pay for a CCXT plan you don't need.

`DataRouter` sits between the agent and adapters and does three things:

1. **Routes by venue prefix.** `NSE:*` goes to Dhan, `DELTA:*` to Delta,
   `BINANCE:*` to CCXT. The agent sees one method call.
2. **Detects gaps.** If you ask for the last 30 days of bars and 25 of
   them are already cached, the router fetches only the 5-day shortfall.
3. **Persists to local storage.** Parquet by default; pluggable to feather
   or pickle when pyarrow isn't available.

A 50-symbol live monitor with naive fetching is thousands of API calls
per day. With the router it's ~50.

## Why immutable run records

Every backtest gets a directory:

```
data/backtests/<run_id>/
    manifest.json     ← strategy, universe, period, params, code_hash
    equity.parquet
    trades.parquet
    stats.json
    log.txt
```

Once written, never modified. Re-running the same params produces the same
`run_id` (it's content-addressed via a SHA-256 of the params dict). Atomic
via a staging directory, so an interrupted write never leaves a half-written
run.

This is what makes "have I already tested this idea?" answerable in six
months. Without immutability, the agent could rewrite history and you'd
never know which version of a result you were looking at.

## Why deflated Sharpe is non-optional

When you sweep parameters, naive Sharpe overstates alpha. The more
configurations you test, the more likely one will look great by chance
alone. By the time the agent has run 200 sweeps, several will look like
gold and waste your time.

`compute_stats(returns, equity, interval, n_trials=N)` returns both raw
Sharpe and the Deflated Sharpe Ratio (Bailey & López de Prado, 2014):

- `dsr > 0.95` — survives multiple-testing correction at the 5% level
- `dsr < 0.95` — plausibly luck given how many configs were tried
- `n_trials=1` reduces this to the Probabilistic Sharpe Ratio (PSR)

The math:

1. Estimate the standard error of the per-period Sharpe using Mertens
   (2002), accounting for skewness and kurtosis in the return
   distribution.
2. Compute the expected maximum Sharpe across `N` random strategies using
   the Bailey-López de Prado closed form.
3. The DSR is the probability that the *true* Sharpe exceeds that
   expected-max threshold, given the observed Sharpe and its standard
   error.

The implementation includes a Beasley-Springer-Moro inverse normal so
SciPy is optional. Tests pin the math against constructed cases —
strong-signal-low-trials passes, weak-signal-high-trials fails — so this
can't silently regress later when someone refactors.

## Why two backtest engines

vectorbt is fast and supports massive parameter sweeps natively. Its fill
model is unrealistic — instant fills at the close, no impact, no order
type complexity.

backtrader is slow and supports realistic limit orders, stop orders,
bracket orders, slippage variance, and per-bar fill modeling. It's what
you want for the final "would this actually work in production"
validation.

The split: **vectorbt for screening, backtrader for validation.** A
strategy that passes vectorbt with `dsr > 0.95` and then survives
backtrader with similar (or only slightly degraded) numbers is something
worth paper trading. A strategy that looks great in vectorbt and falls
apart in backtrader was relying on unrealistic fills.

Both wrappers consume the same `bars_by_symbol` dict from the router
and write the same immutable run record format. They're interchangeable
from the agent's perspective.

## Why SQLite, not Postgres

A solo quant doesn't need a database server. SQLite handles tens of
thousands of backtest records, hundreds of thousands of signal
evaluations, and indefinitely-many memories without breaking a sweat.
One file, easy to back up, easy to inspect with `sqlite3 athena.db`.
When you outgrow it, you'll have years of usage data to inform the
schema migration to Postgres.

The five tables:

| Table                  | Purpose |
| ---                    | --- |
| `backtest_runs`        | Every backtest, indexed for SQL queries |
| `signal_evaluations`   | Every signal/symbol/timestamp triple, with fired/value/context |
| `memories`             | Episodic + semantic memory the agent writes for itself |
| `skill_meta`           | Skill usage counters, last-modified, pinning, archiving |
| `sessions`             | Conversation log scaffolding for the agent loop (Week 3) |

The agent has tools that query these tables directly. "Have I tested
anything similar?" becomes a SQL query the agent writes and runs, not
something it has to remember.

## Why OpenRouter for LLM access

Three roles:

- `smart` — main agent reasoning. Default: Claude Sonnet 4.6.
- `cheap` — background skill review, summarization. Default: Claude Haiku 4.5.
- `embeddings` — semantic memory search. Default: OpenAI text-embedding-3-small.

The cheap-vs-smart split is what makes the self-improvement loop
economical. After every non-trivial turn, a background fork on the cheap
model decides whether to write a memory or skill. At Haiku prices that's
~$5/month. The same loop on Sonnet would be ~$50.

OpenRouter gives you one API key for 200+ models. Switch providers
without code changes. If Anthropic raises prices or drops a model, you
swap to OpenAI or DeepSeek in one config line.

## What's deliberately not in the codebase yet

- **Vector search** for memories. The current `search_memories()` uses
  `LIKE`. Vector search arrives in Week 4 with the knowledge ingestion
  tools, where it earns its keep on volume.
- **WebSocket feeds.** Live signal monitoring on minute bars works fine
  via REST polling for a 50-symbol universe. WebSockets matter for
  sub-second strategies; that's a different tier of system.
- **Trade execution.** Athena alerts; it never trades. Adding execution
  is a deliberate, gated decision — not a default capability.
- **Multi-user anything.** This is a one-person system. No auth, no
  tenancy, no session management beyond what one human needs.

Each of these is a reasonable feature to add later. None of them is
worth building before the core works for one person on one machine.
