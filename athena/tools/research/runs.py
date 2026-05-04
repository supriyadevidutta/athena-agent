"""
Backtest run identity and storage.

Every backtest run is immutable: once written, the directory is never modified.
A new run gets a new run_id. This is what makes "have I tested anything like
this?" answerable six months later.

Layout:
    data/backtests/<run_id>/
        manifest.json     -> strategy, universe, period, params, engine, code_hash
        equity.parquet    -> per-bar equity curve
        trades.parquet    -> trade-level records
        stats.json        -> Sharpe, Sortino, max DD, deflated Sharpe, turnover
        log.txt           -> human-readable summary

run_id scheme:
    YYYYMMDD-HHMMSS-<strategy_slug>-<params_hash8>
    e.g. 20260505-144312-mean_rev_zscore-a3f1c2d8
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:40] or "run"


def params_hash(params: dict) -> str:
    """Stable 8-char hash of a params dict. Same params → same hash, always."""
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def make_run_id(strategy: str, params: dict, ts: Optional[datetime] = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    return f"{ts.strftime('%Y%m%d-%H%M%S')}-{_slug(strategy)}-{params_hash(params)}"


@dataclass
class RunManifest:
    run_id: str
    strategy: str
    engine: str                      # "vectorbt" | "backtrader"
    universe: list[str]              # canonical Athena symbols
    interval: str                    # "1d", "1h", ...
    start: str                       # ISO date
    end: str                         # ISO date
    params: dict
    cost_model: dict                 # fees, slippage assumptions
    code_hash: str                   # SHA-256 of strategy code, if known
    created_at: str                  # ISO timestamp UTC
    notes: str = ""
    parent_run_id: Optional[str] = None  # for sweeps / walk-forward children
    tags: list[str] = field(default_factory=list)


class RunStore:
    """Filesystem-backed registry of backtest runs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, run_id: str) -> Path:
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.path(run_id) / "manifest.json").exists()

    def write(
        self,
        manifest: RunManifest,
        equity: pd.DataFrame,
        trades: pd.DataFrame,
        stats: dict,
        log: str = "",
        cache_format: Optional[str] = None,
    ) -> Path:
        """Write a complete run atomically. Refuses to overwrite an existing run."""
        target = self.path(manifest.run_id)
        if target.exists():
            raise FileExistsError(
                f"run {manifest.run_id} already exists; runs are immutable"
            )
        # Stage in a temp dir, then rename — partial writes never visible.
        staging = target.with_suffix(".staging")
        if staging.exists():
            # leftover from a crash; nuke it
            import shutil
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        fmt = cache_format or os.environ.get("ATHENA_CACHE_FORMAT", "parquet").lower()
        ext = {"parquet": ".parquet", "feather": ".feather", "pickle": ".pkl"}.get(fmt, ".parquet")

        def _write_df(df: pd.DataFrame, path: Path):
            if fmt == "parquet":
                df.to_parquet(path, index=False)
            elif fmt == "feather":
                df.to_feather(path)
            else:
                df.to_pickle(path)

        _write_df(equity, staging / f"equity{ext}")
        _write_df(trades, staging / f"trades{ext}")
        (staging / "manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, default=str)
        )
        (staging / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
        (staging / "log.txt").write_text(log)

        staging.rename(target)
        return target

    def read_manifest(self, run_id: str) -> RunManifest:
        m = json.loads((self.path(run_id) / "manifest.json").read_text())
        return RunManifest(**m)

    def read_stats(self, run_id: str) -> dict:
        return json.loads((self.path(run_id) / "stats.json").read_text())

    def read_equity(self, run_id: str) -> pd.DataFrame:
        d = self.path(run_id)
        for ext, reader in (
            (".parquet", pd.read_parquet),
            (".feather", pd.read_feather),
            (".pkl", pd.read_pickle),
        ):
            fp = d / f"equity{ext}"
            if fp.exists():
                return reader(fp)
        raise FileNotFoundError(f"no equity file for {run_id}")

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and not p.name.endswith(".staging")
                      and (p / "manifest.json").exists())
