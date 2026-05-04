# Quickstart

This walks you through the data spine and research stack end-to-end. It
takes about 15 minutes if you have credentials ready.

## 1. Install

```bash
git clone https://github.com/<your-username>/athena-agent.git
cd athena-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## 2. Configure

Copy the example env file and fill in what you have:

```bash
cp .env.example .env
# Edit .env. For this tutorial you only need OPENROUTER_API_KEY (optional)
# and DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN if you want to test NSE.
```

Or just `export` the variables directly:

```bash
export DHAN_CLIENT_ID="..."
export DHAN_ACCESS_TOKEN="..."
```

You don't need any keys for crypto data — Binance public works without
auth, and Delta Exchange market data works without auth.

## 3. Pull bars from three venues

```python
from datetime import datetime, timezone, timedelta
from athena.tools.data.router import build_default_router

router = build_default_router(
    enable_dhan=True,    # set False if you don't have Dhan creds
    enable_delta=True,
    enable_ccxt=True,
)

end = datetime.now(timezone.utc)
start = end - timedelta(days=90)

# Crypto via Binance public
btc, meta = router.history("BINANCE:BTC/USDT", "1d", start, end)
print(f"BTC bars: {len(btc)}, source: {meta.source}")

# Delta Exchange perpetual
delta_btc, meta = router.history("DELTA:BTCUSD", "1h", start, end)
print(f"Delta BTC bars: {len(delta_btc)}")

# Indian equity
if "NSE" in router.venues():
    rel, meta = router.history("NSE:RELIANCE", "1d", start, end)
    print(f"Reliance bars: {len(rel)}")

# Second call hits the cache — no API quota burned
_, meta = router.history("BINANCE:BTC/USDT", "1d", start, end)
print(f"Second call source: {meta.source}")  # 'cache'
```

## 4. Run a backtest with deflated Sharpe

```python
from athena.tools.research.backtest_vbt import run_vectorbt
from athena.tools.research.runs import RunStore

# Use the bars we already pulled
bars_by_symbol = {
    "BINANCE:BTC/USDT": btc,
    "DELTA:BTCUSD": delta_btc,
}

def momentum_signal(close):
    """Simple 20-day momentum, long when positive."""
    momo = close / close.shift(20) - 1.0
    return (momo > 0), (momo <= 0)

store = RunStore("data/backtests")
result = run_vectorbt(
    strategy="momentum_20d_demo",
    bars_by_symbol=bars_by_symbol,
    signal_fn=momentum_signal,
    interval="1d",
    asset_class="crypto",
    params={"lookback": 20},
    n_trials=1,             # we tested one config
    store=store,
)

print(f"Sharpe:        {result['stats']['sharpe']:.2f}")
print(f"Deflated SR:   {result['stats']['deflated']['dsr']:.2f}")
print(f"Max drawdown: {result['stats']['max_dd']:.2%}")
print(f"Run saved to: {result['path']}")
```

## 5. Sweep parameters and check for false positives

```python
from athena.tools.research.stats import compute_stats
import numpy as np

# Try many lookbacks
results = []
for lookback in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]:
    def make_signal(lb):
        def fn(close):
            momo = close / close.shift(lb) - 1.0
            return (momo > 0), (momo <= 0)
        return fn

    r = run_vectorbt(
        strategy=f"momentum_sweep",
        bars_by_symbol=bars_by_symbol,
        signal_fn=make_signal(lookback),
        interval="1d",
        asset_class="crypto",
        params={"lookback": lookback},
        n_trials=12,            # we're testing 12 configs total
        store=store,
    )
    results.append((lookback, r["stats"]["sharpe"], r["stats"]["deflated"]["dsr"]))

# The best raw Sharpe in the sweep
best = max(results, key=lambda x: x[1])
print(f"Best raw Sharpe: lookback={best[0]}, sharpe={best[1]:.2f}, dsr={best[2]:.2f}")

# The deflated number tells you whether to trust it
if best[2] > 0.95:
    print("✓ Survives multiple-testing correction")
else:
    print("✗ Plausibly luck — don't trust the raw Sharpe alone")
```

## 6. Find prior runs by SQL

```python
from athena.agent.store import Store

s = Store("athena.db")

# Index everything we ran (the run_vectorbt wrapper does this if you
# pass an SQL store — but for this demo we'll do it manually)
for run_id in store.list_runs():
    manifest = store.read_manifest(run_id)
    stats = store.read_stats(run_id)
    s.record_run(manifest, stats)

# Now query
hits = s.find_runs(strategy="momentum_sweep", min_dsr=0.5)
for h in hits:
    print(h["run_id"], h["sharpe"], h["dsr"])
```

## 7. Add a memory the agent will see later

```python
# Find the run_id corresponding to the best lookback
best_run_id = None
for h in hits:
    if abs(h["sharpe"] - best[1]) < 1e-6:
        best_run_id = h["run_id"]
        break

s.add_memory(
    "Tested momentum on BTC perps, lookback sweep 5-120 days. "
    f"Best DSR was at lookback={best[0]}. Worth re-running with realistic costs.",
    kind="episodic",
    tags=["momentum", "btc", "sweep"],
    related_run_id=best_run_id,
)

# Find it later
for m in s.search_memories("momentum"):
    print(m["text"])
```

## What you have now

- Three venues hitting one cache.
- A backtest harness that flags false positives.
- Every run stored immutably and queryable by SQL.
- A memory layer the agent will use in Week 3.

When the agent loop arrives, it will use exactly these primitives. The
manual Python you just wrote becomes the agent's tool calls.
