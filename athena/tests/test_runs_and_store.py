"""
Tests for RunStore + SQLite Store.

Covers:
  - run_id determinism via params hash
  - immutability: re-writing the same run_id raises
  - atomic writes via staging directory (interrupted writes don't corrupt)
  - round-trip: write a run, read manifest/stats/equity back
  - SQL queries: find runs by strategy / sharpe / dsr
  - signal evaluation log + last_eval lookup
  - skill metadata touch and pin/archive flags
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from athena.tools.research.runs import (
    RunManifest, RunStore, make_run_id, params_hash,
)
from athena.agent.store import Store


def _sample_manifest(run_id: str = "") -> RunManifest:
    params = {"lookback": 20, "z_threshold": 2.0}
    rid = run_id or make_run_id("test_strategy", params)
    return RunManifest(
        run_id=rid,
        strategy="test_strategy",
        engine="vectorbt",
        universe=["BINANCE:BTC/USDT", "BINANCE:ETH/USDT"],
        interval="1d",
        start="2024-01-01",
        end="2024-12-31",
        params=params,
        cost_model={"fees": 0.0005, "slippage": 0.0005},
        code_hash="abc123",
        created_at=datetime.now(timezone.utc).isoformat(),
        notes="test run",
        tags=["pairs", "mean-reversion"],
    )


def _sample_equity():
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    return pd.DataFrame({"ts_utc": idx, "equity": [100.0 + i * 0.1 for i in range(50)]})


def _sample_trades():
    return pd.DataFrame({
        "side": ["long", "short"],
        "entry": [100.0, 105.0],
        "exit": [102.0, 103.0],
        "pnl": [2.0, 2.0],
    })


def _sample_stats():
    return {
        "sharpe": 1.5, "sortino": 2.1, "calmar": 0.8,
        "cagr": 0.18, "max_dd": -0.05, "n_obs": 49,
        "deflated": {
            "sharpe": 1.5, "n_trials": 1, "sr_threshold": 0.0,
            "psr": 0.97, "dsr": 0.97, "skew": 0.1, "kurt": 0.2, "n_obs": 49,
        },
    }


# ---------- runs.py --------------------------------------------------------

def test_params_hash_is_stable():
    p1 = {"a": 1, "b": 2}
    p2 = {"b": 2, "a": 1}  # same params, different key order
    assert params_hash(p1) == params_hash(p2)
    p3 = {"a": 1, "b": 3}
    assert params_hash(p1) != params_hash(p3)


def test_run_id_format():
    rid = make_run_id("Mean Reversion!", {"x": 1})
    # YYYYMMDD-HHMMSS-slug-hash8
    parts = rid.split("-")
    assert len(parts) >= 4
    assert len(parts[-1]) == 8


def test_runstore_write_and_read():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = RunStore(tmp)
        m = _sample_manifest()
        store.write(m, _sample_equity(), _sample_trades(), _sample_stats(),
                    log="test", cache_format="pickle")
        assert store.exists(m.run_id)
        m2 = store.read_manifest(m.run_id)
        assert m2.run_id == m.run_id
        assert m2.universe == m.universe
        s = store.read_stats(m.run_id)
        assert abs(s["sharpe"] - 1.5) < 1e-9
        eq = store.read_equity(m.run_id)
        assert len(eq) == 50
    finally:
        shutil.rmtree(tmp)


def test_runstore_refuses_overwrite():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = RunStore(tmp)
        m = _sample_manifest()
        store.write(m, _sample_equity(), _sample_trades(), _sample_stats(),
                    cache_format="pickle")
        # Re-writing same run_id should raise
        try:
            store.write(m, _sample_equity(), _sample_trades(), _sample_stats(),
                        cache_format="pickle")
        except FileExistsError:
            pass
        else:
            raise AssertionError("RunStore allowed overwrite of existing run")
    finally:
        shutil.rmtree(tmp)


def test_runstore_recovers_from_stale_staging():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = RunStore(tmp)
        m = _sample_manifest()
        # Simulate an interrupted previous write — leftover .staging dir
        staging = tmp / (m.run_id + ".staging")
        staging.mkdir()
        (staging / "garbage.txt").write_text("from a crash")
        # New write should clobber the staging and succeed
        store.write(m, _sample_equity(), _sample_trades(), _sample_stats(),
                    cache_format="pickle")
        assert store.exists(m.run_id)
        assert not staging.exists()
    finally:
        shutil.rmtree(tmp)


def test_runstore_list_runs():
    tmp = Path(tempfile.mkdtemp())
    try:
        store = RunStore(tmp)
        for i in range(3):
            m = _sample_manifest(run_id=f"run_{i:03d}")
            store.write(m, _sample_equity(), _sample_trades(), _sample_stats(),
                        cache_format="pickle")
        runs = store.list_runs()
        assert runs == ["run_000", "run_001", "run_002"]
    finally:
        shutil.rmtree(tmp)


# ---------- store.py (SQLite) ---------------------------------------------

def test_store_record_and_find_runs():
    tmp = Path(tempfile.mkdtemp())
    try:
        s = Store(tmp / "athena.db")
        m1 = _sample_manifest("run_a")
        s.record_run(m1, _sample_stats())
        m2 = _sample_manifest("run_b")
        m2 = RunManifest(**{**m1.__dict__, "run_id": "run_b", "strategy": "other"})
        s.record_run(m2, {**_sample_stats(), "sharpe": 0.5})
        # Filter by strategy
        rows = s.find_runs(strategy="test_strategy")
        assert len(rows) == 1 and rows[0]["run_id"] == "run_a"
        # Filter by min Sharpe
        hi = s.find_runs(min_sharpe=1.0)
        assert len(hi) == 1 and hi[0]["run_id"] == "run_a"
        # Filter by min DSR
        d = s.find_runs(min_dsr=0.9)
        assert len(d) == 2
        # Universe round-trip
        import json
        u = json.loads(rows[0]["universe"])
        assert u == m1.universe
    finally:
        shutil.rmtree(tmp)


def test_store_signal_evaluations():
    tmp = Path(tempfile.mkdtemp())
    try:
        s = Store(tmp / "athena.db")
        now = datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)
        s.record_eval("momo", now, "NSE:RELIANCE", fired=True, value=2.3,
                      context={"close": 2900.5})
        s.record_eval("momo", now, "NSE:TCS", fired=False, value=0.4)
        # Last eval lookup
        last = s.last_eval("momo", "NSE:RELIANCE")
        assert last is not None and last["fired"] == 1
        # Recent fires
        fires = s.recent_evals(signal_name="momo", fired_only=True)
        assert len(fires) == 1
        assert fires[0]["symbol"] == "NSE:RELIANCE"
    finally:
        shutil.rmtree(tmp)


def test_store_memories():
    tmp = Path(tempfile.mkdtemp())
    try:
        s = Store(tmp / "athena.db")
        s.add_memory("Mean reversion broke during the RBI announcement on April 12",
                     kind="episodic", tags=["rbi", "vol-shock"])
        s.add_memory("User prefers Kelly-fraction sizing capped at 25%",
                     kind="semantic", tags=["risk"])
        hits = s.search_memories("RBI")
        assert len(hits) == 1
        sem = s.search_memories("Kelly", kind="semantic")
        assert len(sem) == 1
        # Validate kind enforcement
        try:
            s.add_memory("bad", kind="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("add_memory accepted invalid kind")
    finally:
        shutil.rmtree(tmp)


def test_store_skill_metadata():
    tmp = Path(tempfile.mkdtemp())
    try:
        s = Store(tmp / "athena.db")
        s.touch_skill("vectorbt-sweep")
        s.touch_skill("vectorbt-sweep")
        s.touch_skill("realistic-costs")
        skills = s.list_skills()
        # vectorbt-sweep should be first (use_count=2)
        assert skills[0]["skill_name"] == "vectorbt-sweep"
        assert skills[0]["use_count"] == 2
        # Pin a skill
        s.upsert_skill_meta("realistic-costs", pinned=True)
        skills = s.list_skills()
        pinned = [sk for sk in skills if sk["skill_name"] == "realistic-costs"][0]
        assert pinned["pinned"] == 1
        # Archive a skill, default list excludes it
        s.upsert_skill_meta("vectorbt-sweep", archived=True)
        names = [sk["skill_name"] for sk in s.list_skills()]
        assert "vectorbt-sweep" not in names
        names_all = [sk["skill_name"]
                     for sk in s.list_skills(include_archived=True)]
        assert "vectorbt-sweep" in names_all
    finally:
        shutil.rmtree(tmp)


# ---------- Runner ---------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("ATHENA_CACHE_FORMAT", "pickle")
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
