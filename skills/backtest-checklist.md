---
name: backtest-checklist
description: Steps to run before trusting a backtest result
tags: [backtest, process]
pinned: true
---

# Backtest checklist

Before reporting a backtest result as "promising", verify each of these.
The agent should refuse to declare a strategy worth paper trading until
all are green.

## Data hygiene

- [ ] Bars are tz-aware UTC across the full window
- [ ] No NaN OHLC values (the contract guarantees this, but verify if the
      source is unfamiliar)
- [ ] Cache served at least some of the data (sanity check the cache works)
- [ ] No look-ahead: signals computed at bar T use only data through bar T
- [ ] Universe doesn't suffer survivorship bias (live tickers only ≠ live
      tickers as of the test period)

## Statistical rigor

- [ ] `n_trials` passed to `compute_stats` reflects the actual number of
      configurations tested in this sweep, not 1
- [ ] `dsr > 0.95` — survives multiple-testing correction
- [ ] At least 30 observations (DSR is meaningless below this)
- [ ] Skew and kurtosis are reasonable; if `kurt > 10`, check for outlier
      bars that might be data errors

## Cost realism

- [ ] Fees include brokerage + exchange charges + STT (for India) + GST
- [ ] Slippage assumption is at least 1× the average tick size
- [ ] For F&O, lot size and margin are accounted for
- [ ] For crypto, funding rates are subtracted on perp positions held
      overnight

## Cross-engine sanity

- [ ] vectorbt result reproduced in backtrader within 20% Sharpe drift
- [ ] If vectorbt says Sharpe 2.5 and backtrader says 0.3, vectorbt was
      fitting unrealistic fills

## Walk-forward (Week 3+)

- [ ] In-sample / out-of-sample split tested
- [ ] Out-of-sample DSR also > 0.5 (lower bar but still meaningful)

## Memory check

- [ ] Search prior runs: have I tested anything similar?
- [ ] Search memories: have I previously rejected this idea, and why?

If any of these fail, write the failure into a memory before moving on.
The next time the agent considers a similar idea it should see the prior
outcome.
