"""
End-to-end: data router -> signal_fn -> stats -> RunStore -> SQLite.

This test mimics what the agent will do, but without vectorbt (so it runs
in any sandbox). We compute a simple long-flat momentum signal by hand,
synthesize an equity curve, build a RunManifest, persist it, record it
in SQLite, then prove the agent could find it later by SQL alone.

If this passes, the spine is sound.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from athena.tools.data.router import DataRouter
from athena.tools.research.runs import RunManifest, RunStore, make_run_id
from athena.tools.research.stats import compute_stats
from athena.agent.store import Store
from athena.tests.test_data_spine import FakeAdapter


def test_end_to_end_pipeline():
    tmp = Path(tempfile.mkdtemp())
    try:
        # --- Wire up the data spine
        fake = FakeAdapter()
        router = DataRouter({"FAKE": fake}, cache_root=tmp / "cache")

        symbols = ["FAKE:A", "FAKE:B", "FAKE:C"]
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 12, 31, tzinfo=timezone.utc)

        bars_by_symbol = {}
        for sym in symbols:
            df, _meta = router.history(sym, "1d", start, end)
            bars_by_symbol[sym] = df

        # --- Build wide close matrix (mimics what backtest_vbt does internally)
        wide = pd.DataFrame({
            sym: df.set_index("ts_utc")["close"]
            for sym, df in bars_by_symbol.items()
        }).sort_index()

        # --- Trivial signal: 20-day momentum, equal-weight long when positive
        lookback = 20
        momentum = wide / wide.shift(lookback) - 1.0
        weights = (momentum > 0).astype(float)
        # equal-weight across symbols that fired this bar
        row_sums = weights.sum(axis=1).replace(0, np.nan)
        weights = weights.div(row_sums, axis=0).fillna(0.0)

        # Apply weights to next-bar returns (no look-ahead)
        rets = wide.pct_change().shift(-1)
        port_ret = (weights * rets).sum(axis=1).shift(1).dropna()
        port_ret.name = "ret"
        equity = (1 + port_ret).cumprod() * 1_000_000.0
        equity.name = "equity"

        assert len(port_ret) > 100, "not enough returns for stats"

        # --- Stats with a meaningful n_trials (simulating a 50-config sweep)
        stats = compute_stats(
            returns=port_ret,
            equity=equity,
            interval="1d",
            asset_class="equity",
            n_trials=50,
        )
        assert "deflated" in stats and "dsr" in stats["deflated"]

        # --- Persist as a RunStore record
        runstore = RunStore(tmp / "backtests")
        params = {"lookback": lookback, "rule": "momentum_long_flat"}
        run_id = make_run_id("e2e_momentum", params)
        manifest = RunManifest(
            run_id=run_id,
            strategy="e2e_momentum",
            engine="manual",  # we computed by hand for this test
            universe=symbols,
            interval="1d",
            start=str(equity.index.min().date()),
            end=str(equity.index.max().date()),
            params=params,
            cost_model={"fees": 0.0, "slippage": 0.0},
            code_hash="e2e-test",
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=["e2e", "momentum"],
        )
        equity_df = equity.reset_index()
        equity_df.columns = ["ts_utc", "equity"]
        runstore.write(manifest, equity_df, pd.DataFrame(), stats,
                       cache_format="pickle")
        assert runstore.exists(run_id)

        # --- Index in SQLite so the agent can query later
        sql = Store(tmp / "athena.db")
        sql.record_run(manifest, stats)

        # --- Simulate the agent asking: "have I tested any momentum strategies?"
        rows = sql.find_runs(strategy="e2e_momentum")
        assert len(rows) == 1
        assert rows[0]["run_id"] == run_id
        # Universe round-trip
        import json
        assert json.loads(rows[0]["universe"]) == symbols
        assert rows[0]["interval"] == "1d"
        # DSR carried through
        assert rows[0]["dsr"] is not None

        # --- Cache hit on a re-query (zero new adapter calls)
        before = len(fake.history_calls)
        for sym in symbols:
            router.history(sym, "1d", start, end)
        after = len(fake.history_calls)
        assert after == before, f"router re-fetched cached data: {after - before} extra calls"

        # --- Memory write that references the run
        sql.add_memory(
            f"Tested 20-day momentum on {len(symbols)} synthetic symbols; "
            f"sharpe={stats['sharpe']:.2f}, dsr={stats['deflated']['dsr']:.2f}",
            kind="episodic",
            tags=["momentum", "synthetic"],
            related_run_id=run_id,
        )
        hits = sql.search_memories("momentum")
        assert len(hits) == 1
        assert hits[0]["related_run_id"] == run_id

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    os.environ.setdefault("ATHENA_CACHE_FORMAT", "pickle")
    try:
        test_end_to_end_pipeline()
        print("  PASS  test_end_to_end_pipeline")
        print("\n1/1 passed")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  FAIL  test_end_to_end_pipeline: {e}")
        sys.exit(1)
