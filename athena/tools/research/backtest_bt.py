"""
backtrader wrapper. Used for production validation of strategies that
vectorbt has already greenlit during exploration.

backtrader gives us realistic order types (limit, stop, bracket), per-bar
fill modeling, slippage variance, and broker simulation that vectorbt
hand-waves. The cost is speed — backtrader is ~100x slower. So we use
vectorbt for sweeps, backtrader for the final "would this actually work"
validation before paper trading.

Contract:
    inputs:
        bars_by_symbol: dict[symbol -> DataFrame in BARS_COLUMNS]
        strategy_cls: a backtrader.Strategy subclass (the user/agent writes this)
        params: dict passed to strategy_cls
        cost_model: {"commission": float, "slippage_perc": float}
        initial_cash: float
    output:
        Run record with run_id, stats, manifest, equity, trades.
"""
from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timezone
from typing import Optional, Type

import numpy as np
import pandas as pd

from .runs import RunManifest, RunStore, make_run_id
from .stats import compute_stats


def _code_hash(cls: Type) -> str:
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        src = repr(cls)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def run_backtrader(
    *,
    strategy: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy_cls,                     # backtrader.Strategy subclass
    interval: str,
    asset_class: str = "equity",
    params: Optional[dict] = None,
    cost_model: Optional[dict] = None,
    initial_cash: float = 1_000_000.0,
    n_trials: int = 1,
    store: Optional[RunStore] = None,
    notes: str = "",
    tags: Optional[list[str]] = None,
    parent_run_id: Optional[str] = None,
) -> dict:
    """Run a backtrader backtest and (optionally) persist it."""
    try:
        import backtrader as bt  # type: ignore
    except ImportError as e:
        raise ImportError(
            "backtrader not installed. pip install backtrader"
        ) from e

    params = dict(params or {})
    cost_model = dict(cost_model or {"commission": 0.0005, "slippage_perc": 0.0005})

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.set_cash(initial_cash)
    cerebro.broker.setcommission(commission=cost_model["commission"])
    # Slippage as percentage of fill price
    cerebro.broker.set_slippage_perc(perc=cost_model["slippage_perc"])

    # Add data feeds
    for sym, df in bars_by_symbol.items():
        feed_df = df.set_index("ts_utc").copy()
        feed_df.index = pd.to_datetime(feed_df.index, utc=True).tz_convert(None)
        # backtrader expects: open, high, low, close, volume, openinterest
        feed_df = feed_df.rename(columns={"oi": "openinterest"})
        feed_df = feed_df[["open", "high", "low", "close", "volume", "openinterest"]]
        feed = bt.feeds.PandasData(dataname=feed_df, name=sym)
        cerebro.adddata(feed, name=sym)

    cerebro.addstrategy(strategy_cls, **params)

    # Analyzers — pick the right TimeFrame for the interval so the equity
    # curve has matching granularity.
    bt_timeframe_map = {
        "1m": bt.TimeFrame.Minutes,
        "5m": bt.TimeFrame.Minutes,
        "15m": bt.TimeFrame.Minutes,
        "1h": bt.TimeFrame.Minutes,
        "1d": bt.TimeFrame.Days,
    }
    bt_compression_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1}
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="timereturn",
        timeframe=bt_timeframe_map.get(interval, bt.TimeFrame.Days),
        compression=bt_compression_map.get(interval, 1),
    )
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    # Equity curve from broker
    timereturn = strat.analyzers.timereturn.get_analysis()
    if timereturn:
        idx = pd.to_datetime(list(timereturn.keys()), utc=True)
        rets = pd.Series(list(timereturn.values()), index=idx, name="ret")
        equity = (1 + rets).cumprod() * initial_cash
        equity.name = "equity"
    else:
        equity = pd.Series([initial_cash],
                           index=pd.to_datetime([datetime.now(timezone.utc)]),
                           name="equity")
        rets = pd.Series([], name="ret", dtype=float)

    # Trades — backtrader's TradeAnalyzer is a nested dict; flatten the closed trades.
    trade_records = []
    try:
        ta = strat.analyzers.trades.get_analysis()
        # Pull aggregate-level info; per-trade records require a custom observer,
        # which we deliberately skip in this wrapper to keep it lean.
        trade_records.append({
            "total_closed": ta.get("total", {}).get("closed", 0),
            "won": ta.get("won", {}).get("total", 0),
            "lost": ta.get("lost", {}).get("total", 0),
            "pnl_net": ta.get("pnl", {}).get("net", {}).get("total", 0),
        })
    except Exception:
        pass
    trades_df = pd.DataFrame(trade_records)

    stats = compute_stats(
        returns=rets,
        equity=equity,
        interval=interval,
        asset_class=asset_class,
        n_trials=n_trials,
    )

    equity_df = equity.reset_index()
    equity_df.columns = ["ts_utc", "equity"]

    universe = sorted(bars_by_symbol.keys())
    start = equity.index.min().date().isoformat() if len(equity) else ""
    end = equity.index.max().date().isoformat() if len(equity) else ""

    full_params = {**params, "n_trials": n_trials, "initial_cash": initial_cash}
    run_id = make_run_id(strategy, full_params)
    manifest = RunManifest(
        run_id=run_id,
        strategy=strategy,
        engine="backtrader",
        universe=universe,
        interval=interval,
        start=start,
        end=end,
        params=full_params,
        cost_model=cost_model,
        code_hash=_code_hash(strategy_cls),
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
        parent_run_id=parent_run_id,
        tags=list(tags or []),
    )

    log = (
        f"strategy={strategy} engine=backtrader\n"
        f"universe={universe}\n"
        f"period={start} .. {end} interval={interval}\n"
        f"sharpe={stats['sharpe']:.3f} cagr={stats['cagr']:.3%} "
        f"max_dd={stats['max_dd']:.3%}\n"
        f"deflated: dsr={stats['deflated']['dsr']:.3f}\n"
    )

    written_path = None
    if store is not None:
        written_path = store.write(manifest, equity_df, trades_df, stats, log)

    return {
        "run_id": run_id,
        "manifest": manifest,
        "stats": stats,
        "equity": equity_df,
        "trades": trades_df,
        "path": str(written_path) if written_path else None,
        "log": log,
    }
