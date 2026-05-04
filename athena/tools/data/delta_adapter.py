"""
Delta Exchange adapter. Uses public REST endpoints — no auth needed for
market data. Delta supports spot, perpetuals, futures, and options on
crypto, all from one venue.

API reference: https://docs.delta.exchange
Base URL: https://api.delta.exchange

Delta product types we map:
  spot         -> asset_class=crypto
  perpetual    -> asset_class=future  (perpetual; expiry=None)
  futures      -> asset_class=future
  call_options -> asset_class=option, right=C
  put_options  -> asset_class=option, right=P
"""
from __future__ import annotations

from datetime import datetime, timezone, date
from typing import Optional, Any
import time

import pandas as pd

from .contract import (
    BARS_COLUMNS, CHAIN_COLUMNS, AdapterError, Instrument, Meta,
    NotSupported, Quote, RateLimited, SymbolNotFound, validate_bars,
)
from .symbols import build, parse


_BASE = "https://api.delta.exchange"

# Delta interval names -> Athena
_INTERVAL_TO_DELTA = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d",
}
_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400,
}


class DeltaAdapter:
    venue = "DELTA"
    asset_classes = ("crypto", "future", "option")

    def __init__(self, base_url: str = _BASE, timeout: float = 10.0):
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise AdapterError("requests not installed. pip install requests") from e
        import requests
        self._requests = requests
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._products: Optional[dict[str, dict]] = None  # cache: symbol -> product

    # ---- HTTP -----------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self._base}{path}"
        try:
            r = self._requests.get(url, params=params, timeout=self._timeout)
        except self._requests.RequestException as e:
            raise AdapterError(f"delta http error: {e}") from e
        if r.status_code == 429:
            raise RateLimited("delta 429")
        if not r.ok:
            raise AdapterError(f"delta {r.status_code}: {r.text[:200]}")
        return r.json()

    def _load_products(self) -> dict[str, dict]:
        if self._products is not None:
            return self._products
        data = self._get("/v2/products")
        if not data.get("success"):
            raise AdapterError(f"delta /products failed: {data}")
        out: dict[str, dict] = {}
        for p in data.get("result", []):
            out[p["symbol"]] = p
        self._products = out
        return out

    # ---- Resolution -----------------------------------------------------

    def _athena_to_delta(self, symbol: str) -> tuple[str, dict]:
        """Map an Athena canonical symbol to a Delta product symbol + product dict."""
        p = parse(symbol)
        if p.venue != self.venue:
            raise SymbolNotFound(f"{symbol} not on DELTA")
        products = self._load_products()

        # Direct: DELTA:BTCUSD -> "BTCUSD" if it exists
        if not p.is_option and not p.is_future:
            if p.root in products:
                return p.root, products[p.root]
            raise SymbolNotFound(f"no DELTA product named {p.root}")

        # Options: Delta uses "C-BTC-80000-270626" or "P-BTC-..."
        if p.is_option:
            assert p.expiry and p.strike and p.right
            # Delta date: DDMMYY
            d = p.expiry.strftime("%d%m%y")
            strike = f"{p.strike:g}"
            cand = f"{p.right}-{p.root}-{strike}-{d}"
            if cand in products:
                return cand, products[cand]
            raise SymbolNotFound(f"no DELTA option matching {symbol} "
                                 f"(tried {cand})")

        # Futures (dated)
        if p.is_future:
            # Delta dated futures look like "BTCUSD_DDMMYY" historically;
            # perpetuals are just "BTCUSD". We try both.
            if p.expiry is None:
                if p.root in products:
                    return p.root, products[p.root]
            else:
                d = p.expiry.strftime("%d%m%y")
                cand = f"{p.root}_{d}"
                if cand in products:
                    return cand, products[cand]
            raise SymbolNotFound(f"no DELTA future matching {symbol}")

        raise SymbolNotFound(symbol)

    def _classify(self, product: dict) -> str:
        ct = (product.get("contract_type") or "").lower()
        if "option" in ct:
            return "option"
        if "future" in ct or "perpetual" in ct:
            return "future"
        return "crypto"

    # ---- Interface ------------------------------------------------------

    def search(self, query: str) -> list[Instrument]:
        products = self._load_products()
        q = query.upper()
        out = []
        for sym, prod in products.items():
            if q not in sym.upper():
                continue
            ac = self._classify(prod)
            out.append(Instrument(
                symbol=build(self.venue, sym),
                asset_class=ac,  # type: ignore[arg-type]
                venue=self.venue,
                name=prod.get("description", sym),
                ccy=prod.get("settling_asset", {}).get("symbol", ""),
                tick_size=float(prod.get("tick_size") or 0),
                lot_size=int(prod.get("contract_value") or 1) if ac != "option" else 1,
                multiplier=float(prod.get("contract_value") or 1),
                extra={"product_id": prod.get("id"),
                       "contract_type": prod.get("contract_type")},
            ))
            if len(out) >= 50:
                break
        return out

    def history(self, symbol: str, interval: str,
                start: datetime, end: datetime) -> tuple[pd.DataFrame, Meta]:
        if interval not in _INTERVAL_TO_DELTA:
            raise NotSupported(f"interval {interval} not supported")
        delta_sym, prod = self._athena_to_delta(symbol)
        resolution = _INTERVAL_TO_DELTA[interval]

        start_s = int(start.timestamp())
        end_s = int(end.timestamp())

        # Delta caps history per call; chunk on resolution * 2000 bars.
        chunk = _INTERVAL_SECONDS[interval] * 2000
        rows: list[dict] = []
        cursor = start_s
        while cursor < end_s:
            stop = min(cursor + chunk, end_s)
            data = self._get("/v2/history/candles", {
                "symbol": delta_sym,
                "resolution": resolution,
                "start": cursor,
                "end": stop,
            })
            if not data.get("success"):
                raise AdapterError(f"delta candles error: {data}")
            batch = data.get("result", []) or []
            if not batch:
                cursor = stop
                continue
            rows.extend(batch)
            last_t = max(int(r["time"]) for r in batch)
            if last_t <= cursor:
                cursor = stop
            else:
                cursor = last_t + 1
            time.sleep(0.05)  # be polite

        if not rows:
            df = pd.DataFrame(columns=BARS_COLUMNS)
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        else:
            df = pd.DataFrame(rows)
            # Delta candle fields: time, open, high, low, close, volume
            df["ts_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["symbol"] = symbol
            df["asset_class"] = self._classify(prod)
            df["venue"] = self.venue
            df["oi"] = 0.0  # candles endpoint doesn't carry OI
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[BARS_COLUMNS]
            df = (df.dropna(subset=["open", "high", "low", "close"])
                  .drop_duplicates(subset=["ts_utc"])
                  .sort_values("ts_utc")
                  .reset_index(drop=True))
        validate_bars(df)

        meta = Meta(
            instrument=Instrument(
                symbol=symbol,
                asset_class=self._classify(prod),  # type: ignore[arg-type]
                venue=self.venue,
                name=delta_sym,
            ),
            interval=interval,
            source="delta:rest",
            retrieved_at=datetime.now(timezone.utc),
            rows=len(df),
        )
        return df, meta

    def quote(self, symbol: str) -> Quote:
        delta_sym, _ = self._athena_to_delta(symbol)
        data = self._get(f"/v2/tickers/{delta_sym}")
        if not data.get("success"):
            raise AdapterError(f"delta ticker error: {data}")
        t = data["result"]
        ts = t.get("timestamp")
        # Delta timestamps are microseconds in some endpoints
        if ts and ts > 1e14:
            ts_utc = datetime.fromtimestamp(ts / 1e6, tz=timezone.utc)
        elif ts:
            ts_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        else:
            ts_utc = datetime.now(timezone.utc)
        return Quote(
            ts_utc=ts_utc,
            symbol=symbol,
            venue=self.venue,
            bid=float(t.get("quotes", {}).get("best_bid") or t.get("best_bid") or 0),
            ask=float(t.get("quotes", {}).get("best_ask") or t.get("best_ask") or 0),
            last=float(t.get("close") or t.get("mark_price") or t.get("spot_price") or 0),
            volume=float(t.get("volume") or 0),
        )

    def chain(self, underlying: str,
              expiry: Optional[date] = None) -> tuple[pd.DataFrame, Meta]:
        """Return the live options chain for an underlying.

        underlying: Athena root, e.g. 'BTC' or 'ETH'. (Bare root, not BTCUSD.)
        expiry: filter to one expiry, or None for all listed.
        """
        products = self._load_products()
        chain_rows: list[dict] = []
        for sym, prod in products.items():
            ct = (prod.get("contract_type") or "").lower()
            if "option" not in ct:
                continue
            # Delta options carry underlying info on the product
            ua = prod.get("underlying_asset", {}).get("symbol")
            if ua != underlying.upper():
                continue
            # parse expiry from settlement_time
            settlement = prod.get("settlement_time")
            try:
                exp_dt = datetime.fromisoformat(settlement.replace("Z", "+00:00"))
                exp_d = exp_dt.date()
            except Exception:
                continue
            if expiry is not None and exp_d != expiry:
                continue
            strike = float(prod.get("strike_price") or 0)
            right = "C" if ct.startswith("call") else "P"
            # quote
            try:
                t = self._get(f"/v2/tickers/{sym}").get("result", {})
            except AdapterError:
                t = {}
            chain_rows.append({
                "ts_utc": pd.Timestamp.now(tz="UTC"),
                "underlying": underlying.upper(),
                "expiry": exp_d,
                "strike": strike,
                "right": right,
                "venue": self.venue,
                "bid": float(t.get("quotes", {}).get("best_bid") or 0),
                "ask": float(t.get("quotes", {}).get("best_ask") or 0),
                "last": float(t.get("close") or t.get("mark_price") or 0),
                "volume": float(t.get("volume") or 0),
                "oi": float(t.get("oi") or 0),
                "iv": float(t.get("mark_iv") or t.get("quotes", {}).get("ask_iv") or 0),
            })

        df = pd.DataFrame(chain_rows, columns=CHAIN_COLUMNS)
        meta = Meta(
            instrument=Instrument(
                symbol=build(self.venue, underlying),
                asset_class="option",
                venue=self.venue,
                name=f"{underlying} chain",
            ),
            source="delta:rest",
            retrieved_at=datetime.now(timezone.utc),
            rows=len(df),
        )
        return df, meta
