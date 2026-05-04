"""
vectorbt wrapper. Optimized for parameter sweeps and exploration.

Contract:
    inputs:
        bars_by_symbol: dict[symbol -> DataFrame in BARS_COLUMNS]
        signal_fn: (close: DataFrame) -> (entries: DataFrame, exits: DataFrame)
                   takes a wide DataFrame [ts_utc x symbols] of close prices,
                   returns boolean entries and exits of the same shape.
        params: dict (recorded in manifest, not interpreted here)
        cost_model: dict {"fees": float, "slippage": float}
                    fees and slippage are fractions, e.g. 0.0005 = 5 bps
    output:
        Run record: equity, trades, stats. Written to RunStore.

Why a single signal_fn instead of full strategy classes:
    The agent will write signal functions inline in skill files. Keeping the
    contract tiny means a skill is just "here's the signal_fn, here's the
    universe, here's the param sweep" — no boilerplate.
"""
from __future__ import annotations

import hashlib
import inspect
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .runs import RunManifest, RunStore, make_run_id
from .stats import compute_stats


SignalFn = Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]


def _wide_close(bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pivot bar frames to a wide [ts_utc x symbols] close-price DataFrame.
    Frames may have different lengths; we outer-join on ts_utc and forward-fill
    nothing — leaving NaNs where data is missing keeps the strategy honest.
    """
    series = {}
    for sym, df in bars_by_symbol.items():
        s = df.set_index("ts_utc")["close"]
        s.index = pd.to_datetime(s.index, utc=True)
        series[sym] = s
    wide = pd.DataFrame(series).sort_index()
    return wide


def _code_hash(fn: Callable) -> str:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def run_vectorbt(
    *,
    strategy: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    signal_fn: SignalFn,
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
    """Run a vectorbt backtest and (optionally) persist it.

    Returns a dict with run_id, stats, manifest, and the equity/trades frames
    (so the caller can use them without round-tripping through disk).
    """
    try:
        import vectorbt as vbt  # type: ignore
    except ImportError as e:
        raise ImportError(
            "vectorbt not installed. Add to your venv: pip install vectorbt"
        ) from e

    params = dict(params or {})
    cost_model = dict(cost_model or {"fees": 0.0005, "slippage": 0.0005})

    close = _wide_close(bars_by_symbol)
    if close.empty:
        raise ValueError("no bars provided")

    entries, exits = signal_fn(close)
    # Reindex to be safe in case signal_fn returns a different shape
    entries = entries.reindex_like(close).fillna(False).astype(bool)
    exits = exits.reindex_like(close).fillna(False).astype(bool)

    # vectorbt freq from interval (pandas 2.x+ requires lowercase units)
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min",
                "1h": "1h", "1d": "1D"}
    freq = freq_map.get(interval, "1D")

    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        freq=freq,
        init_cash=initial_cash,
        fees=cost_model["fees"],
        slippage=cost_model["slippage"],
        # Equal weight across symbols by default; cash_sharing flattens this.
        cash_sharing=True,
        group_by=True,
    )

    # Aggregate equity & returns
    equity_curve = portfolio.value()
    if isinstance(equity_curve, pd.DataFrame):
        # When grouped, vectorbt may still return a DataFrame; reduce to Series
        equity_curve = equity_curve.sum(axis=1)
    equity_curve.name = "equity"
    equity_curve.index = pd.to_datetime(equity_curve.index, utc=True)

    returns = equity_curve.pct_change().dropna()
    returns.name = "ret"

    stats = compute_stats(
        returns=returns,
        equity=equity_curve,
        interval=interval,
        asset_class=asset_class,
        n_trials=n_trials,
    )

    # Trades DataFrame
    try:
        trades_df = portfolio.trades.records_readable.copy()
    except Exception:
        trades_df = pd.DataFrame()
    if not trades_df.empty:
        trades_df.columns = [c.lower().replace(" ", "_") for c in trades_df.columns]

    equity_df = equity_curve.reset_index()
    equity_df.columns = ["ts_utc", "equity"]

    # Build manifest
    universe = sorted(bars_by_symbol.keys())
    start = equity_curve.index.min().date().isoformat() if len(equity_curve) else ""
    end = equity_curve.index.max().date().isoformat() if len(equity_curve) else ""

    full_params = {**params, "n_trials": n_trials,
                   "initial_cash": initial_cash}
    run_id = make_run_id(strategy, full_params)
    manifest = RunManifest(
        run_id=run_id,
        strategy=strategy,
        engine="vectorbt",
        universe=universe,
        interval=interval,
        start=start,
        end=end,
        params=full_params,
        cost_model=cost_model,
        code_hash=_code_hash(signal_fn),
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=notes,
        parent_run_id=parent_run_id,
        tags=list(tags or []),
    )

    log = (
        f"strategy={strategy} engine=vectorbt\n"
        f"universe={universe}\n"
        f"period={start} .. {end} interval={interval}\n"
        f"sharpe={stats['sharpe']:.3f} cagr={stats['cagr']:.3%} "
        f"max_dd={stats['max_dd']:.3%}\n"
        f"deflated: dsr={stats['deflated']['dsr']:.3f} "
        f"sr_threshold={stats['deflated']['sr_threshold']:.3f}\n"
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
