"""
Dhan adapter. Uses Dhan HQ API v2.
Reference: https://dhanhq.co/docs/v2/

Auth: requires DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN env vars.
Endpoints used:
  POST /v2/charts/historical   -> daily bars
  POST /v2/charts/intraday     -> intraday bars
  POST /v2/marketfeed/quote    -> live quotes
  GET  /v2/optionchain         -> options chain (some plans)

Dhan identifies instruments by `securityId` (an integer per exchange segment).
We resolve Athena symbols (e.g. NSE:RELIANCE, NSE:NIFTY:20260529:F) to
securityId via Dhan's instrument master CSV, which we cache locally.
"""
from __future__ import annotations

import csv
import io
import os
import time
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, Any

import pandas as pd

from .contract import (
    BARS_COLUMNS, CHAIN_COLUMNS, AdapterError, Instrument, Meta,
    NotSupported, Quote, RateLimited, SymbolNotFound, validate_bars,
)
from .symbols import build, parse


_BASE = "https://api.dhan.co"
_INSTRUMENT_CSV = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

# Dhan exchange segment codes
_SEG = {
    "NSE_EQ": "NSE_EQ",
    "BSE_EQ": "BSE_EQ",
    "NSE_FNO": "NSE_FNO",
    "BSE_FNO": "BSE_FNO",
    "MCX_COMM": "MCX_COMM",
    "IDX_I": "IDX_I",     # indices
}

# Dhan intraday interval (minutes as int)
_INTRADAY_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}


class DhanAdapter:
    venue = "NSE"  # default; we also serve BSE/MCX through the same adapter
    asset_classes = ("equity", "future", "option", "commodity")

    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        base_url: str = _BASE,
        instrument_cache: Optional[Path] = None,
        timeout: float = 15.0,
    ):
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise AdapterError("requests not installed. pip install requests") from e
        import requests
        self._requests = requests
        self._client_id = client_id or os.environ.get("DHAN_CLIENT_ID", "")
        self._token = access_token or os.environ.get("DHAN_ACCESS_TOKEN", "")
        if not self._client_id or not self._token:
            raise AdapterError(
                "Dhan credentials missing. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN."
            )
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._inst_cache_path = (
            instrument_cache
            or Path.home() / ".athena" / "dhan_instruments.csv"
        )
        self._instruments: Optional[pd.DataFrame] = None

    # ---- HTTP -----------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "access-token": self._token,
            "client-id": self._client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self._base}{path}"
        try:
            r = self._requests.post(url, json=body,
                                    headers=self._headers(),
                                    timeout=self._timeout)
        except self._requests.RequestException as e:
            raise AdapterError(f"dhan http error: {e}") from e
        if r.status_code == 429:
            raise RateLimited("dhan 429")
        if not r.ok:
            raise AdapterError(f"dhan {r.status_code}: {r.text[:200]}")
        return r.json()

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self._base}{path}"
        try:
            r = self._requests.get(url, params=params,
                                   headers=self._headers(),
                                   timeout=self._timeout)
        except self._requests.RequestException as e:
            raise AdapterError(f"dhan http error: {e}") from e
        if r.status_code == 429:
            raise RateLimited("dhan 429")
        if not r.ok:
            raise AdapterError(f"dhan {r.status_code}: {r.text[:200]}")
        return r.json()

    # ---- Instrument master ---------------------------------------------

    def _load_instruments(self, force: bool = False) -> pd.DataFrame:
        if self._instruments is not None and not force:
            return self._instruments
        # Refresh if older than 24h
        stale = True
        if self._inst_cache_path.exists() and not force:
            age = time.time() - self._inst_cache_path.stat().st_mtime
            stale = age > 86400
        if stale:
            self._inst_cache_path.parent.mkdir(parents=True, exist_ok=True)
            r = self._requests.get(_INSTRUMENT_CSV, timeout=60)
            if not r.ok:
                if self._inst_cache_path.exists():
                    pass  # fall back to stale cache
                else:
                    raise AdapterError(
                        f"failed to fetch Dhan instrument master: {r.status_code}"
                    )
            else:
                self._inst_cache_path.write_bytes(r.content)
        df = pd.read_csv(self._inst_cache_path, low_memory=False)
        # Normalize column names
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        self._instruments = df
        return df

    def _resolve(self, symbol: str) -> tuple[int, str, dict]:
        """Return (security_id, exchange_segment, instrument_row)."""
        p = parse(symbol)
        df = self._load_instruments()

        def _col(name: str) -> pd.Series:
            """Return column or an empty-string series of right length."""
            if name in df.columns:
                return df[name]
            return pd.Series([""] * len(df), index=df.index)

        if p.venue == "NSE" and not p.is_option and not p.is_future:
            # Equity or index
            ts_col = "SEM_TRADING_SYMBOL" if "SEM_TRADING_SYMBOL" in df.columns \
                     else "SEM_CUSTOM_SYMBOL"
            mask = (_col(ts_col) == p.root) & (_col("SEM_EXM_EXCH_ID") == "NSE")
            sub = df[mask]
            # Prefer EQ over indices when ambiguous
            if "SEM_SERIES" in sub.columns:
                eq = sub[sub["SEM_SERIES"].isin(["EQ", "BE"])]
                if len(eq):
                    sub = eq
            if len(sub) == 0:
                # Index fallback
                idx_mask = ((_col("SEM_TRADING_SYMBOL") == p.root) &
                            (_col("SEM_EXM_EXCH_ID") == "NSE"))
                sub = df[idx_mask]
            if len(sub) == 0:
                raise SymbolNotFound(f"NSE:{p.root} not in instrument master")
            row = sub.iloc[0].to_dict()
            seg = (row.get("SEM_SEGMENT") or "").upper()
            seg_map = {"E": "NSE_EQ", "I": "IDX_I", "D": "NSE_FNO"}
            segment = seg_map.get(seg, "NSE_EQ")
            return int(row["SEM_SMST_SECURITY_ID"]), segment, row

        if p.venue == "NSE" and (p.is_option or p.is_future):
            # F&O: match by root + expiry + (strike, right) for options
            mask = (
                _col("SEM_INSTRUMENT_NAME").astype(str)
                    .str.contains(p.root, case=False, na=False)
                & (_col("SEM_EXM_EXCH_ID") == "NSE")
                & (_col("SEM_SEGMENT").astype(str).str.upper() == "D")
            )
            sub = df[mask]
            if "SEM_EXPIRY_DATE" in sub.columns and p.expiry is not None:
                sub = sub[pd.to_datetime(sub["SEM_EXPIRY_DATE"],
                                         errors="coerce").dt.date == p.expiry]
            if p.is_option:
                if "SEM_OPTION_TYPE" in sub.columns:
                    sub = sub[sub["SEM_OPTION_TYPE"].astype(str).str.upper()
                              == ("CE" if p.right == "C" else "PE")]
                if "SEM_STRIKE_PRICE" in sub.columns:
                    sub = sub[pd.to_numeric(sub["SEM_STRIKE_PRICE"],
                                            errors="coerce") == p.strike]
            else:  # future
                if "SEM_OPTION_TYPE" in sub.columns:
                    sub = sub[sub["SEM_OPTION_TYPE"].astype(str).str.upper()
                              .isin(["XX", "FF", ""])]
            if len(sub) == 0:
                raise SymbolNotFound(f"could not resolve {symbol} in F&O master")
            row = sub.iloc[0].to_dict()
            return int(row["SEM_SMST_SECURITY_ID"]), "NSE_FNO", row

        raise SymbolNotFound(f"venue {p.venue} not implemented in DhanAdapter")

    def _instrument_type(self, segment: str, row: dict, p) -> str:
        """Map (segment, instrument-master row, parsed symbol) to Dhan's
        `instrument` field for the charts endpoints.

        Dhan accepts: EQUITY, INDEX, FUTSTK, FUTIDX, FUTCOM, OPTSTK, OPTIDX, OPTFUT.
        """
        if segment == "NSE_EQ" or segment == "BSE_EQ":
            return "EQUITY"
        if segment == "IDX_I":
            return "INDEX"
        if segment == "MCX_COMM":
            return "FUTCOM"
        # F&O
        underlying_seg = (row.get("SEM_INSTRUMENT_NAME") or "").upper()
        if p.is_option:
            if "IDX" in underlying_seg or underlying_seg.startswith("OPTIDX"):
                return "OPTIDX"
            return "OPTSTK"
        if p.is_future:
            if "IDX" in underlying_seg or underlying_seg.startswith("FUTIDX"):
                return "FUTIDX"
            return "FUTSTK"
        return "EQUITY"

    # ---- Interface ------------------------------------------------------

    def search(self, query: str) -> list[Instrument]:
        df = self._load_instruments()
        q = query.upper()
        col = ("SEM_TRADING_SYMBOL" if "SEM_TRADING_SYMBOL" in df.columns
               else "SEM_CUSTOM_SYMBOL")
        mask = df[col].astype(str).str.upper().str.contains(q, na=False)
        sub = df[mask].head(50)
        out = []
        for _, row in sub.iterrows():
            seg = (row.get("SEM_SEGMENT") or "").upper()
            ac = "equity" if seg == "E" else "option" if seg == "D" else "equity"
            out.append(Instrument(
                symbol=build("NSE", row[col]),
                asset_class=ac,  # type: ignore[arg-type]
                venue="NSE",
                name=str(row.get("SEM_CUSTOM_SYMBOL") or row[col]),
                ccy="INR",
                tick_size=float(row.get("SEM_TICK_SIZE") or 0),
                lot_size=int(row.get("SEM_LOT_UNITS") or 1),
            ))
        return out

    def history(self, symbol: str, interval: str,
                start: datetime, end: datetime) -> tuple[pd.DataFrame, Meta]:
        sec_id, segment, row = self._resolve(symbol)
        p = parse(symbol)
        instrument = self._instrument_type(segment, row, p)
        if interval == "1d":
            body = {
                "securityId": str(sec_id),
                "exchangeSegment": segment,
                "instrument": instrument,
                "fromDate": start.date().isoformat(),
                "toDate": end.date().isoformat(),
            }
            data = self._post("/v2/charts/historical", body)
        elif interval in _INTRADAY_MIN:
            body = {
                "securityId": str(sec_id),
                "exchangeSegment": segment,
                "instrument": instrument,
                "interval": str(_INTRADAY_MIN[interval]),
                "fromDate": start.date().isoformat(),
                "toDate": end.date().isoformat(),
            }
            data = self._post("/v2/charts/intraday", body)
        else:
            raise NotSupported(f"interval {interval} not supported")

        # Dhan returns parallel arrays: timestamp[], open[], high[], low[],
        # close[], volume[], (open_interest[] for FNO)
        ts = data.get("timestamp") or data.get("start_Time") or []
        opens = data.get("open") or []
        highs = data.get("high") or []
        lows = data.get("low") or []
        closes = data.get("close") or []
        vols = data.get("volume") or [0] * len(ts)
        ois = data.get("open_interest") or [0] * len(ts)

        # Determine asset_class from the parsed symbol + segment
        if p.is_option:
            ac = "option"
        elif p.is_future:
            ac = "future"
        elif segment == "IDX_I":
            ac = "equity"  # treat indices as equity for downstream stats
        elif segment == "MCX_COMM":
            ac = "commodity"
        else:
            ac = "equity"

        if not ts:
            df = pd.DataFrame(columns=BARS_COLUMNS)
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        else:
            df = pd.DataFrame({
                "ts_utc": pd.to_datetime(ts, unit="s", utc=True),
                "symbol": symbol,
                "asset_class": ac,
                "venue": "NSE",
                "open": opens, "high": highs, "low": lows, "close": closes,
                "volume": vols, "oi": ois,
            })[BARS_COLUMNS]
            df = (df.dropna(subset=["open", "high", "low", "close"])
                  .drop_duplicates(subset=["ts_utc"])
                  .sort_values("ts_utc")
                  .reset_index(drop=True))
        validate_bars(df)

        meta = Meta(
            instrument=Instrument(
                symbol=symbol,
                asset_class=ac,  # type: ignore[arg-type]
                venue="NSE",
                name=str(row.get("SEM_CUSTOM_SYMBOL") or row.get("SEM_TRADING_SYMBOL") or ""),
                ccy="INR",
                tick_size=float(row.get("SEM_TICK_SIZE") or 0),
                lot_size=int(row.get("SEM_LOT_UNITS") or 1),
            ),
            interval=interval,
            source="dhan:v2",
            retrieved_at=datetime.now(timezone.utc),
            rows=len(df),
        )
        return df, meta

    def quote(self, symbol: str) -> Quote:
        sec_id, segment, _ = self._resolve(symbol)
        body = {segment: [int(sec_id)]}
        data = self._post("/v2/marketfeed/quote", body)
        # Response shape: {data: {SEGMENT: {sec_id: {...}}}}
        try:
            entry = data["data"][segment][str(sec_id)]
        except (KeyError, TypeError):
            try:
                entry = data["data"][segment][int(sec_id)]
            except Exception as e:
                raise AdapterError(f"unexpected quote shape: {data}") from e
        depth = entry.get("depth", {})
        bid = float((depth.get("buy") or [{}])[0].get("price") or 0)
        ask = float((depth.get("sell") or [{}])[0].get("price") or 0)
        return Quote(
            ts_utc=datetime.now(timezone.utc),
            symbol=symbol,
            venue="NSE",
            bid=bid,
            ask=ask,
            last=float(entry.get("last_price") or entry.get("LTP") or 0),
            volume=float(entry.get("volume") or 0),
        )

    def chain(self, underlying: str,
              expiry: Optional[date] = None) -> tuple[pd.DataFrame, Meta]:
        # Dhan's option chain endpoint requires an underlying scrip ID.
        sec_id, _, row = self._resolve(f"NSE:{underlying}")
        body = {
            "UnderlyingScrip": int(sec_id),
            "UnderlyingSeg": "IDX_I" if (row.get("SEM_SEGMENT") or "").upper() == "I"
                             else "NSE_EQ",
        }
        if expiry is not None:
            body["Expiry"] = expiry.isoformat()
        data = self._post("/v2/optionchain", body)
        oc = data.get("data", {}).get("oc", {})
        rows = []
        now = pd.Timestamp.now(tz="UTC")
        for strike_str, leg in oc.items():
            try:
                strike = float(strike_str)
            except ValueError:
                continue
            for right_key, right in (("ce", "C"), ("pe", "P")):
                d = leg.get(right_key) or {}
                if not d:
                    continue
                rows.append({
                    "ts_utc": now,
                    "underlying": underlying.upper(),
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "venue": "NSE",
                    "bid": float(d.get("top_bid_price") or 0),
                    "ask": float(d.get("top_ask_price") or 0),
                    "last": float(d.get("last_price") or 0),
                    "volume": float(d.get("volume") or 0),
                    "oi": float(d.get("oi") or 0),
                    "iv": float(d.get("implied_volatility") or 0),
                })
        df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
        meta = Meta(
            instrument=Instrument(
                symbol=build("NSE", underlying),
                asset_class="option", venue="NSE",
                name=f"{underlying} chain", ccy="INR",
            ),
            source="dhan:v2",
            retrieved_at=datetime.now(timezone.utc),
            rows=len(df),
        )
        return df, meta
