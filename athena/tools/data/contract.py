"""
Athena data contract. Every adapter speaks this dialect.
The agent and downstream tools never see vendor-specific shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Protocol, runtime_checkable, Optional, Literal

import pandas as pd


AssetClass = Literal["equity", "future", "option", "crypto", "fx", "commodity"]
Interval = Literal["1m", "5m", "15m", "1h", "1d"]


# ---------- Schemas ---------------------------------------------------------

BARS_COLUMNS = ["ts_utc", "symbol", "asset_class", "venue",
                "open", "high", "low", "close", "volume", "oi"]

QUOTE_COLUMNS = ["ts_utc", "symbol", "venue", "bid", "ask", "last", "volume"]

CHAIN_COLUMNS = ["ts_utc", "underlying", "expiry", "strike", "right",
                 "venue", "bid", "ask", "last", "volume", "oi", "iv"]


# ---------- Value objects ---------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    symbol: str          # canonical Athena symbol, e.g. "NSE:RELIANCE", "DELTA:BTCUSD"
    asset_class: AssetClass
    venue: str
    name: str = ""
    ccy: str = ""
    tick_size: float = 0.0
    lot_size: int = 1
    multiplier: float = 1.0
    expiry: Optional[date] = None
    strike: Optional[float] = None
    right: Optional[str] = None  # "C" / "P" for options
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Meta:
    """Travels alongside a DataFrame; never lives inside it."""
    instrument: Instrument
    interval: Optional[str] = None
    source: str = ""
    retrieved_at: Optional[datetime] = None
    rows: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.retrieved_at is not None:
            d["retrieved_at"] = self.retrieved_at.isoformat()
        if self.instrument.expiry is not None:
            d["instrument"]["expiry"] = self.instrument.expiry.isoformat()
        return d


@dataclass(frozen=True)
class Quote:
    ts_utc: datetime
    symbol: str
    venue: str
    bid: float
    ask: float
    last: float
    volume: float = 0.0


# ---------- Errors ----------------------------------------------------------

class AdapterError(Exception):
    """Base for adapter-layer errors."""

class NotSupported(AdapterError):
    """The adapter doesn't implement this method for this asset class."""

class SymbolNotFound(AdapterError):
    """Canonical symbol couldn't be resolved at the venue."""

class RateLimited(AdapterError):
    """Vendor rate limit hit. Backoff is the caller's job."""


# ---------- Protocol --------------------------------------------------------

@runtime_checkable
class DataAdapter(Protocol):
    """The four methods every adapter must offer.

    Adapters may raise NotSupported for any method that doesn't apply
    (e.g., CCXT public endpoints don't have most options chains).
    """
    venue: str
    asset_classes: tuple[AssetClass, ...]

    def search(self, query: str) -> list[Instrument]: ...

    def history(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, Meta]: ...

    def quote(self, symbol: str) -> Quote: ...

    def chain(
        self,
        underlying: str,
        expiry: Optional[date] = None,
    ) -> tuple[pd.DataFrame, Meta]: ...


# ---------- Validation ------------------------------------------------------

def validate_bars(df: pd.DataFrame) -> None:
    """Raise if a DataFrame doesn't conform to the bars contract.
    Adapters call this on their way out; tests call it on the way in.
    """
    missing = set(BARS_COLUMNS) - set(df.columns)
    if missing:
        raise AdapterError(f"bars frame missing columns: {sorted(missing)}")
    if len(df) == 0:
        return
    ts = df["ts_utc"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise AdapterError("ts_utc must be datetime64")
    # tz-aware UTC required
    tz = getattr(ts.dt, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise AdapterError(f"ts_utc must be tz-aware UTC, got tz={tz}")
    if not df["ts_utc"].is_monotonic_increasing:
        raise AdapterError("ts_utc must be monotonically increasing")
    for col in ("open", "high", "low", "close"):
        if df[col].isna().any():
            raise AdapterError(f"{col} contains NaN")
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        raise AdapterError(f"{bad_hl} rows have high < low")
