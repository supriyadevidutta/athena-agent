"""
Tests for the data spine.

Uses a FakeAdapter that returns synthetic bars — no network calls, fast,
deterministic. If these pass, the contract + cache + router are sound.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

import pandas as pd
import numpy as np

# Make athena imports work whether run from repo root or tests/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from athena.tools.data.contract import (
    BARS_COLUMNS, AdapterError, Instrument, Meta, NotSupported, Quote,
    validate_bars,
)
from athena.tools.data import cache as cache_mod
from athena.tools.data.router import DataRouter
from athena.tools.data.symbols import build, parse


# ---------- Fake adapter ----------------------------------------------------

class FakeAdapter:
    venue = "FAKE"
    asset_classes = ("equity",)

    def __init__(self):
        self.history_calls: list[tuple] = []

    def search(self, query):
        return [Instrument(symbol=build("FAKE", "TEST"),
                           asset_class="equity", venue="FAKE")]

    def history(self, symbol, interval, start, end):
        self.history_calls.append((symbol, interval, start, end))
        # Generate a deterministic walk
        def _ts(dt):
            t = pd.Timestamp(dt)
            return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
        s = _ts(start).floor("D")
        e = _ts(end).floor("D")
        idx = pd.date_range(s, e, freq="D", tz="UTC")
        n = len(idx)
        rng = np.random.default_rng(seed=int(s.timestamp()) % (2**32))
        rets = rng.normal(0, 0.01, n)
        close = 100.0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "ts_utc": idx,
            "symbol": symbol,
            "asset_class": "equity",
            "venue": "FAKE",
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1000, 10000, n).astype(float),
            "oi": 0.0,
        })[BARS_COLUMNS]
        # Ensure high>=max(o,c), low<=min(o,c)
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        validate_bars(df)
        meta = Meta(
            instrument=Instrument(symbol=symbol, asset_class="equity", venue="FAKE"),
            interval=interval, source="fake",
            retrieved_at=datetime.now(timezone.utc), rows=len(df),
        )
        return df, meta

    def quote(self, symbol):
        return Quote(ts_utc=datetime.now(timezone.utc), symbol=symbol,
                     venue="FAKE", bid=100.0, ask=100.5, last=100.25)

    def chain(self, underlying, expiry=None):
        raise NotSupported("FakeAdapter has no options")


# ---------- Tests -----------------------------------------------------------

def test_symbol_parse_build_roundtrip():
    cases = [
        "NSE:RELIANCE",
        "DELTA:BTCUSD",
        "BINANCE:BTC/USDT",
    ]
    for s in cases:
        p = parse(s)
        assert build(p.venue, p.root) == s, f"{s} != roundtrip"
    # Option
    s = "NSE:NIFTY:20260529:24500:C"
    p = parse(s)
    assert p.is_option and p.right == "C" and p.strike == 24500
    assert build("NSE", "NIFTY", expiry=p.expiry, strike=24500, right="C") == s
    # Future
    s = "NSE:NIFTY:20260529:F"
    p = parse(s)
    assert p.is_future and p.expiry == date(2026, 5, 29)


def test_validate_bars_rejects_bad_frames():
    # missing column
    bad = pd.DataFrame({"ts_utc": pd.to_datetime(["2024-01-01"], utc=True),
                        "symbol": ["X"]})
    try:
        validate_bars(bad)
    except AdapterError as e:
        assert "missing columns" in str(e)
    else:
        raise AssertionError("expected AdapterError for missing columns")

    # naive datetime
    df = pd.DataFrame({c: [] for c in BARS_COLUMNS})
    df.loc[0] = [pd.Timestamp("2024-01-01"), "X", "equity", "FAKE",
                 1.0, 2.0, 0.5, 1.5, 100.0, 0.0]
    try:
        validate_bars(df)
    except AdapterError as e:
        assert "tz-aware" in str(e) or "UTC" in str(e)
    else:
        raise AssertionError("expected AdapterError for naive ts_utc")


def test_cache_roundtrip_and_dedup():
    tmp = Path(tempfile.mkdtemp())
    try:
        adapter = FakeAdapter()
        sym = "FAKE:TEST"
        df1, _ = adapter.history(sym, "1d",
                                 datetime(2024, 1, 1, tzinfo=timezone.utc),
                                 datetime(2024, 1, 10, tzinfo=timezone.utc))
        cache_mod.write(tmp, sym, "1d", df1)
        # Overlapping fetch
        df2, _ = adapter.history(sym, "1d",
                                 datetime(2024, 1, 5, tzinfo=timezone.utc),
                                 datetime(2024, 1, 15, tzinfo=timezone.utc))
        merged = cache_mod.upsert(tmp, sym, "1d", df2)
        # Dedup: no duplicate timestamps
        assert merged["ts_utc"].is_unique, "cache failed to dedup"
        # Coverage spans the union
        cov = cache_mod.coverage(tmp, sym, "1d")
        assert cov is not None
        assert cov[0].date() == date(2024, 1, 1)
        assert cov[1].date() == date(2024, 1, 15)
        # Round-trip preserves UTC tz
        back = cache_mod.read(tmp, sym, "1d")
        assert back is not None
        assert str(back["ts_utc"].dt.tz) == "UTC"
    finally:
        shutil.rmtree(tmp)


def test_router_serves_from_cache_on_second_call():
    tmp = Path(tempfile.mkdtemp())
    try:
        fake = FakeAdapter()
        router = DataRouter({"FAKE": fake}, cache_root=tmp)
        sym = "FAKE:TEST"
        s = datetime(2024, 1, 1, tzinfo=timezone.utc)
        e = datetime(2024, 1, 10, tzinfo=timezone.utc)
        df1, m1 = router.history(sym, "1d", s, e)
        assert len(df1) > 0
        assert len(fake.history_calls) == 1

        # Same window again — should be served from cache, no new adapter call
        df2, m2 = router.history(sym, "1d", s, e)
        assert len(fake.history_calls) == 1, "router re-fetched a cached range"
        assert m2.source == "cache"
        # Frames equal
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))
    finally:
        shutil.rmtree(tmp)


def test_router_extends_cache_on_right_edge():
    tmp = Path(tempfile.mkdtemp())
    try:
        fake = FakeAdapter()
        router = DataRouter({"FAKE": fake}, cache_root=tmp)
        sym = "FAKE:TEST"
        # First fetch: 10 days
        router.history(sym, "1d",
                       datetime(2024, 1, 1, tzinfo=timezone.utc),
                       datetime(2024, 1, 10, tzinfo=timezone.utc))
        # Second fetch: extends to day 20 — should fetch only the gap
        df2, _ = router.history(sym, "1d",
                                datetime(2024, 1, 1, tzinfo=timezone.utc),
                                datetime(2024, 1, 20, tzinfo=timezone.utc))
        assert len(fake.history_calls) == 2
        # Second call's start should be at or after Jan 10 (the cached right edge)
        second_start = fake.history_calls[1][2]
        assert second_start >= datetime(2024, 1, 10, tzinfo=timezone.utc)
        # Returned union spans full window
        assert df2["ts_utc"].iloc[0] <= pd.Timestamp("2024-01-02", tz="UTC")
        assert df2["ts_utc"].iloc[-1] >= pd.Timestamp("2024-01-19", tz="UTC")
    finally:
        shutil.rmtree(tmp)


def test_router_routes_by_venue():
    fake = FakeAdapter()
    router = DataRouter({"FAKE": fake}, cache_root=Path(tempfile.mkdtemp()))
    q = router.quote("FAKE:TEST")
    assert q.bid == 100.0


def test_router_raises_for_unknown_venue():
    router = DataRouter({"FAKE": FakeAdapter()},
                        cache_root=Path(tempfile.mkdtemp()))
    try:
        router.quote("UNKNOWN:FOO")
    except Exception as e:
        assert "no adapter" in str(e).lower()
    else:
        raise AssertionError("expected SymbolNotFound for unknown venue")


# ---------- Runner ---------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
