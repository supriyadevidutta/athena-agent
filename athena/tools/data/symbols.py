"""
Canonical symbol scheme: VENUE:RAW or VENUE:UNDERLYING:YYYYMMDD:STRIKE:R

Examples:
    NSE:RELIANCE                       Indian equity
    NSE:NIFTY:20260529:24500:C         NIFTY call option
    NSE:NIFTY:20260529:F               NIFTY future
    DELTA:BTCUSD                       Delta perpetual / spot
    DELTA:BTCUSD:20260627:80000:C      Delta dated option
    BINANCE:BTC/USDT                   CCXT spot

Adapters parse and emit only this form. Vendor-specific tokens stay
inside the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class ParsedSymbol:
    venue: str
    root: str
    expiry: Optional[date] = None
    strike: Optional[float] = None
    right: Optional[str] = None  # "C", "P", or "F" for futures

    @property
    def is_option(self) -> bool:
        return self.right in ("C", "P")

    @property
    def is_future(self) -> bool:
        return self.right == "F"


def parse(symbol: str) -> ParsedSymbol:
    parts = symbol.split(":")
    if len(parts) < 2:
        raise ValueError(f"symbol must contain venue prefix: {symbol!r}")
    venue, root, *rest = parts
    if not rest:
        return ParsedSymbol(venue=venue, root=root)
    # future: VENUE:ROOT:YYYYMMDD:F
    if len(rest) == 2 and rest[1] == "F":
        return ParsedSymbol(
            venue=venue, root=root,
            expiry=date.fromisoformat(_iso(rest[0])),
            right="F",
        )
    # option: VENUE:ROOT:YYYYMMDD:STRIKE:R
    if len(rest) == 3:
        return ParsedSymbol(
            venue=venue, root=root,
            expiry=date.fromisoformat(_iso(rest[0])),
            strike=float(rest[1]),
            right=rest[2].upper(),
        )
    raise ValueError(f"unrecognized symbol shape: {symbol!r}")


def build(venue: str, root: str, *,
          expiry: Optional[date] = None,
          strike: Optional[float] = None,
          right: Optional[str] = None) -> str:
    if right == "F" and expiry is not None:
        return f"{venue}:{root}:{expiry.strftime('%Y%m%d')}:F"
    if right in ("C", "P") and expiry is not None and strike is not None:
        s = f"{strike:g}"
        return f"{venue}:{root}:{expiry.strftime('%Y%m%d')}:{s}:{right}"
    return f"{venue}:{root}"


def _iso(yyyymmdd: str) -> str:
    if len(yyyymmdd) != 8:
        raise ValueError(f"expiry must be YYYYMMDD: {yyyymmdd!r}")
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
