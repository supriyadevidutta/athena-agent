"""
DataRouter — single entry point for the agent and tools.

The agent never instantiates an adapter directly. It calls router.history(),
router.quote(), router.chain(). Routing is driven by the venue prefix in the
canonical Athena symbol.

The router also handles the cache-and-gap logic so adapters stay dumb.
"""
from __future__ import annotations

from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .contract import (
    AdapterError, DataAdapter, Instrument, Meta, NotSupported, Quote,
    SymbolNotFound,
)
from . import cache
from .symbols import parse


def _to_utc_ts(dt: datetime) -> pd.Timestamp:
    """Coerce a possibly-naive datetime to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class DataRouter:
    def __init__(
        self,
        adapters: dict[str, DataAdapter],
        cache_root: Optional[Path] = None,
    ):
        self._adapters = adapters
        self._cache_root = cache_root or (Path.home() / ".athena" / "cache")
        self._cache_root.mkdir(parents=True, exist_ok=True)

    # ---- Discovery ------------------------------------------------------

    def venues(self) -> list[str]:
        return sorted(self._adapters.keys())

    def _adapter_for(self, symbol: str) -> DataAdapter:
        venue = parse(symbol).venue
        if venue not in self._adapters:
            raise SymbolNotFound(
                f"no adapter registered for venue {venue!r}. "
                f"Available: {self.venues()}"
            )
        return self._adapters[venue]

    # ---- Pass-through ---------------------------------------------------

    def search(self, query: str, venue: Optional[str] = None):
        if venue:
            return self._adapters[venue].search(query)
        out = []
        for a in self._adapters.values():
            try:
                out.extend(a.search(query))
            except Exception:
                continue
        return out

    def quote(self, symbol: str) -> Quote:
        return self._adapter_for(symbol).quote(symbol)

    def chain(self, symbol: str, expiry: Optional[date] = None):
        adapter = self._adapter_for(symbol)
        try:
            return adapter.chain(parse(symbol).root, expiry)
        except NotSupported as e:
            raise NotSupported(
                f"{adapter.venue} doesn't expose chains for {symbol}: {e}"
            ) from e

    # ---- History (with cache) -------------------------------------------

    def history(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> tuple[pd.DataFrame, Meta]:
        adapter = self._adapter_for(symbol)
        if not use_cache:
            return adapter.history(symbol, interval, start, end)

        gap = cache.gap(self._cache_root, symbol, interval, start, end)
        if gap is None:
            df = cache.read(self._cache_root, symbol, interval)
            assert df is not None
            sliced = df[(df["ts_utc"] >= _to_utc_ts(start)) &
                        (df["ts_utc"] <= _to_utc_ts(end))].reset_index(drop=True)
            p = parse(symbol)
            meta = Meta(
                instrument=Instrument(
                    symbol=symbol,
                    asset_class=sliced["asset_class"].iloc[0] if len(sliced) else "equity",
                    venue=p.venue,
                ),
                interval=interval,
                source="cache",
                retrieved_at=datetime.now(timezone.utc),
                rows=len(sliced),
                notes="served fully from cache",
            )
            return sliced, meta

        gap_start, gap_end = gap
        new_df, meta = adapter.history(symbol, interval, gap_start, gap_end)
        if len(new_df) > 0:
            cache.upsert(self._cache_root, symbol, interval, new_df)
        full = cache.read(self._cache_root, symbol, interval)
        if full is None:
            return new_df, meta
        sliced = full[(full["ts_utc"] >= _to_utc_ts(start)) &
                      (full["ts_utc"] <= _to_utc_ts(end))].reset_index(drop=True)
        meta = Meta(
            instrument=meta.instrument,
            interval=interval,
            source=meta.source + "+cache",
            retrieved_at=meta.retrieved_at,
            rows=len(sliced),
            notes=f"gap={gap_start.isoformat()}..{gap_end.isoformat()}",
        )
        return sliced, meta


def build_default_router(
    enable_dhan: bool = False,
    enable_delta: bool = True,
    enable_ccxt: bool = True,
    cache_root: Optional[Path] = None,
) -> DataRouter:
    """Wire up the adapters you have credentials for. Skips ones that fail to init."""
    adapters: dict[str, DataAdapter] = {}
    if enable_ccxt:
        try:
            from .ccxt_adapter import CCXTAdapter
            a = CCXTAdapter("binance", venue="BINANCE")
            adapters[a.venue] = a
        except Exception as e:
            print(f"[router] CCXT adapter skipped: {e}")
    if enable_delta:
        try:
            from .delta_adapter import DeltaAdapter
            a = DeltaAdapter()
            adapters[a.venue] = a
        except Exception as e:
            print(f"[router] Delta adapter skipped: {e}")
    if enable_dhan:
        try:
            from .dhan_adapter import DhanAdapter
            a = DhanAdapter()
            adapters[a.venue] = a
        except Exception as e:
            print(f"[router] Dhan adapter skipped: {e}")
    return DataRouter(adapters, cache_root=cache_root)
