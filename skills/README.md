# Skills

This directory holds markdown files the agent reads and writes during
operation. Each skill is a single `.md` file documenting a procedure the
agent has either been taught or learned from experience.

## Anatomy of a skill

```markdown
---
name: realistic-costs
description: Cost assumptions per asset class for backtests
tags: [backtest, costs]
last_used: 2026-05-05
---

# Realistic costs

For NSE F&O:
- Brokerage: ₹20 per executed order (Dhan flat fee)
- STT: 0.0125% on sell side for futures, 0.0625% on options sell premium
- ...

For crypto on Binance:
- Maker: 0.10%, taker: 0.10% by default
- ...
```

## How they're used (Week 3+)

When the agent gets a task, it searches this directory for relevant
skills and loads matching ones into the system prompt before reasoning.
After completing a non-trivial task, a background fork on the cheap
model decides whether to write a new skill or patch an existing one.

## Manual seeds

You can seed skills by hand. Good first ones to write:

- `risk-sizing.md` — your position sizing rules
- `realistic-costs.md` — fee and slippage assumptions per venue
- `data-quality-gotchas.md` — known issues with each data source
- `backtest-checklist.md` — the steps you always run before trusting a result

## Don't commit user-specific skills

By default, `.gitignore` excludes everything in this directory except
this README. Skills are personal to your research and probably shouldn't
go in version control. If you want to share a generic skill template
with the project, add an explicit `!skills/<name>.md` line to
`.gitignore`.
