"""
CCXT adapter for crypto. Public endpoints only — no keys required for market
data. Use Binance by default; CCXT lets us swap exchanges with one line.

We deliberately don't implement chain() here — most CCXT exchanges don't expose
options chains uniformly. Use the Delta adapter for crypto options.
"""
from __future__ import annotations

from datetime import datetime, timezone, date
from typing import Optional

import pandas as pd

from .contract import (
    BARS_COLUMNS, AdapterError, Instrument, Meta, NotSupported,
    Quote, RateLimited, SymbolNotFound, validate_bars,
)
from .symbols import build, parse

# CCXT interval map
_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d",
}


class CCXTAdapter:
    venue = "BINANCE"
    asset_classes = ("crypto",)

    def __init__(self, exchange_id: str = "binance", venue: Optional[str] = None):
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise AdapterError(
                "ccxt not installed. pip install ccxt"
            ) from e
        self._ccxt = ccxt
        self._ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
        if venue:
            self.venue = venue
        else:
            self.venue = exchange_id.upper()

    # ---- Resolution ------------------------------------------------------

    def _to_ccxt_symbol(self, symbol: str) -> str:
        p = parse(symbol)
        if p.venue != self.venue:
            raise SymbolNotFound(f"{symbol} not on venue {self.venue}")
        # Athena root for crypto is "BTC/USDT" or "BTCUSDT"; CCXT wants "BTC/USDT"
        root = p.root
        if "/" not in root:
            # try common splits
            for q in ("USDT", "USDC", "USD", "BTC", "ETH"):
                if root.endswith(q):
                    root = f"{root[:-len(q)]}/{q}"
                    break
        return root

    # ---- Interface -------------------------------------------------------

    def search(self, query: str) -> list[Instrument]:
        self._ex.load_markets()
        q = query.upper()
        out = []
        for sym, m in self._ex.markets.items():
            if q in sym.upper():
                out.append(Instrument(
                    symbol=build(self.venue, sym),
                    asset_class="crypto",
                    venue=self.venue,
                    name=sym,
                    ccy=m.get("quote", ""),
                    tick_size=float(m.get("precision", {}).get("price", 0) or 0),
                    lot_size=1,
                    multiplier=1.0,
                ))
                if len(out) >= 50:
                    break
        return out

    def history(self, symbol: str, interval: str,
                start: datetime, end: datetime) -> tuple[pd.DataFrame, Meta]:
        if interval not in _INTERVAL_MAP:
            raise NotSupported(f"interval {interval} not supported")
        ccxt_sym = self._to_ccxt_symbol(symbol)

        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_rows: list[list] = []
        cursor = since_ms
        # CCXT returns up to ~1000 bars per call; loop until past end_ms.
        while cursor < end_ms:
            try:
                rows = self._ex.fetch_ohlcv(
                    ccxt_sym, _INTERVAL_MAP[interval], since=cursor, limit=1000,
                )
            except self._ccxt.RateLimitExceeded as e:
                raise RateLimited(str(e)) from e
            except self._ccxt.BaseError as e:
                raise AdapterError(f"ccxt error: {e}") from e
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            if last_ts <= cursor:  # no progress; bail
                break
            cursor = last_ts + 1

        if not all_rows:
            df = pd.DataFrame(columns=BARS_COLUMNS)
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        else:
            df = pd.DataFrame(all_rows,
                              columns=["ts_ms", "open", "high", "low", "close", "volume"])
            df["ts_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            df["symbol"] = symbol
            df["asset_class"] = "crypto"
            df["venue"] = self.venue
            df["oi"] = 0.0
            df = df[BARS_COLUMNS]
            df = (df[(df["ts_utc"] >= pd.Timestamp(start, tz="UTC")) &
                     (df["ts_utc"] <= pd.Timestamp(end, tz="UTC"))]
                  .drop_duplicates(subset=["ts_utc"])
                  .sort_values("ts_utc")
                  .reset_index(drop=True))
        validate_bars(df)

        meta = Meta(
            instrument=Instrument(
                symbol=symbol, asset_class="crypto",
                venue=self.venue, name=ccxt_sym,
            ),
            interval=interval,
            source=f"ccxt:{self._ex.id}",
            retrieved_at=datetime.now(timezone.utc),
            rows=len(df),
        )
        return df, meta

    def quote(self, symbol: str) -> Quote:
        ccxt_sym = self._to_ccxt_symbol(symbol)
        t = self._ex.fetch_ticker(ccxt_sym)
        return Quote(
            ts_utc=datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc)
                   if t.get("timestamp") else datetime.now(timezone.utc),
            symbol=symbol,
            venue=self.venue,
            bid=float(t.get("bid") or 0),
            ask=float(t.get("ask") or 0),
            last=float(t.get("last") or 0),
            volume=float(t.get("baseVolume") or 0),
        )

    def chain(self, underlying: str,
              expiry: Optional[date] = None) -> tuple[pd.DataFrame, Meta]:
        raise NotSupported("CCXT public adapter doesn't expose options chains. "
                           "Use the Delta adapter for crypto options.")
