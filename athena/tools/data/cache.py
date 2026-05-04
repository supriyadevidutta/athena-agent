"""
Local cache for bars. Partitioned by venue/interval/symbol.

Cache is content-addressed by (symbol, interval) — each file holds the full
history we've fetched. On every history() call the adapter checks the cache,
fetches only the gap, appends, deduplicates, and rewrites.

This is the single biggest cost reducer in the whole system. Live signal
monitoring on a 50-symbol universe with naive fetching = thousands of API
calls per day. With cache = ~50.

Storage backend is pluggable: parquet in production (set ATHENA_CACHE_FORMAT=parquet
and install pyarrow), feather as a fallback. Both preserve dtype and tz.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .contract import BARS_COLUMNS, validate_bars
from .symbols import parse


def _format() -> str:
    return os.environ.get("ATHENA_CACHE_FORMAT", "parquet").lower()


def _ext() -> str:
    fmt = _format()
    return {"parquet": ".parquet", "feather": ".feather", "pickle": ".pkl"}.get(fmt, ".parquet")


def _path(root: Path, symbol: str, interval: str) -> Path:
    p = parse(symbol)
    safe = symbol.replace(":", "_").replace("/", "-")
    return root / p.venue / interval / f"{safe}{_ext()}"


def _read_file(fp: Path) -> pd.DataFrame:
    fmt = _format()
    if fmt == "parquet":
        return pd.read_parquet(fp)
    if fmt == "feather":
        return pd.read_feather(fp)
    return pd.read_pickle(fp)


def _write_file(df: pd.DataFrame, fp: Path) -> None:
    fmt = _format()
    if fmt == "parquet":
        df.to_parquet(fp, index=False)
    elif fmt == "feather":
        df.to_feather(fp)
    else:
        df.to_pickle(fp)


def read(root: Path, symbol: str, interval: str) -> Optional[pd.DataFrame]:
    fp = _path(root, symbol, interval)
    if not fp.exists():
        return None
    df = _read_file(fp)
    # Round-trip can drop tz; restore.
    if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    elif df["ts_utc"].dt.tz is None:
        df["ts_utc"] = df["ts_utc"].dt.tz_localize("UTC")
    return df


def write(root: Path, symbol: str, interval: str, df: pd.DataFrame) -> None:
    validate_bars(df)
    fp = _path(root, symbol, interval)
    fp.parent.mkdir(parents=True, exist_ok=True)
    _write_file(df, fp)


def upsert(root: Path, symbol: str, interval: str,
           new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new bars into the cache, dedupe on ts_utc, return the union."""
    validate_bars(new_df)
    existing = read(root, symbol, interval)
    if existing is None or len(existing) == 0:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = (merged
                  .drop_duplicates(subset=["ts_utc"], keep="last")
                  .sort_values("ts_utc")
                  .reset_index(drop=True))
    write(root, symbol, interval, merged)
    return merged


def coverage(root: Path, symbol: str, interval: str
             ) -> Optional[tuple[datetime, datetime]]:
    df = read(root, symbol, interval)
    if df is None or len(df) == 0:
        return None
    return df["ts_utc"].iloc[0].to_pydatetime(), df["ts_utc"].iloc[-1].to_pydatetime()


def gap(root: Path, symbol: str, interval: str,
        start: datetime, end: datetime) -> Optional[tuple[datetime, datetime]]:
    """Return the (sub)range we still need to fetch, or None if cache covers it.

    Naive but works: if cache covers [a,b], and we want [s,e]:
      - if b >= e and a <= s: nothing to fetch
      - else fetch the union shortfall (we keep it simple — fetch [min(s,a), max(e,b)]
        bounded by what's missing on the right edge, which is the common case for
        live monitoring)
    """
    cov = coverage(root, symbol, interval)
    if cov is None:
        return _ensure_utc(start), _ensure_utc(end)
    a, b = cov
    s, e = _ensure_utc(start), _ensure_utc(end)
    if a <= s and b >= e:
        return None
    if b < e and a <= s:
        # right-edge extension, the common live case
        return b, e
    # else fetch the whole requested range and let upsert merge
    return s, e


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
