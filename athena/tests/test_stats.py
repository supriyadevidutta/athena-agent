"""
Tests for athena.tools.research.stats.

These tests pin the math, especially the deflated Sharpe — that calculation
is the most consequential one in the whole research stack and it must not
silently regress.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))

from athena.tools.research.stats import (
    sharpe, sortino, max_drawdown, calmar, cagr, hit_rate, turnover,
    deflated_sharpe, compute_stats, bars_per_year, _expected_max_z, _inv_norm,
)


def _const_returns(mu_per_day: float, n: int = 504, seed: int = 42) -> pd.Series:
    """Returns with known mean, zero noise. Sharpe is then trivially computable."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series([mu_per_day] * n, index=idx, name="ret")


def _normal_returns(mu: float, sigma: float, n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(rng.normal(mu, sigma, n), index=idx, name="ret")


# ---------- Sharpe and friends ---------------------------------------------

def test_sharpe_zero_for_zero_mean():
    r = _normal_returns(mu=0.0, sigma=0.01, n=1000)
    s = sharpe(r, periods_per_year=252)
    # Should be small, not exactly zero (sample noise), but well within ~0.3
    assert abs(s) < 0.5, f"sharpe of zero-mean returns drifted: {s}"


def test_sharpe_scales_with_periods_per_year():
    r = _normal_returns(mu=0.001, sigma=0.01, n=1000)
    s_daily = sharpe(r, periods_per_year=252)
    s_hourly = sharpe(r, periods_per_year=252 * 6.5)
    # Higher annualization factor → higher annualized Sharpe by sqrt ratio
    assert s_hourly > s_daily * 1.5


def test_sharpe_handles_constant_returns():
    # All identical returns → zero std → NaN
    r = _const_returns(0.001, n=100)
    s = sharpe(r, periods_per_year=252)
    assert math.isnan(s)


def test_sortino_lower_when_downside_skew():
    # Symmetric: Sortino ≈ Sharpe * sqrt(2)
    r = _normal_returns(mu=0.001, sigma=0.01, n=2000)
    s = sharpe(r, 252)
    so = sortino(r, 252)
    assert so > s, "sortino should exceed sharpe for symmetric returns"


def test_max_drawdown_finds_correct_trough():
    # Engineered series: up to 100, down to 50, back up
    eq = pd.Series(
        [100, 110, 120, 90, 50, 70, 100, 130],
        index=pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"),
    )
    mdd, peak, trough = max_drawdown(eq)
    # Peak 120 → trough 50, dd = (50-120)/120 ≈ -0.5833
    assert abs(mdd - (50 - 120) / 120) < 1e-9
    assert peak == eq.index[2]
    assert trough == eq.index[4]


def test_max_drawdown_zero_for_monotonic():
    eq = pd.Series([100, 110, 120, 130],
                   index=pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"))
    mdd, _, _ = max_drawdown(eq)
    assert mdd == 0.0


def test_cagr_recovers_known_growth():
    # Start 100, end 121 over 1 year (252 daily bars + 1) → ~21% CAGR
    n = 253
    growth = (121 / 100) ** (1 / 252)
    eq = pd.Series([100 * (growth ** i) for i in range(n)],
                   index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"))
    g = cagr(eq, 252)
    assert abs(g - 0.21) < 0.01


def test_hit_rate():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.005],
                  name="ret")
    # nonzero: [0.01, -0.005, 0.02, -0.01, 0.005] → 3 positives / 5 = 0.6
    assert abs(hit_rate(r) - 0.6) < 1e-9


def test_turnover():
    pos = pd.Series([0, 1, 1, 0, -1, -1, 0])
    # diffs: [_, 1, 0, 1, 1, 0, 1] → mean abs = 4/6
    assert abs(turnover(pos) - 4 / 6) < 1e-9


# ---------- Deflated Sharpe -------------------------------------------------

def test_inv_norm_basic_quantiles():
    assert abs(_inv_norm(0.5)) < 1e-6
    assert abs(_inv_norm(0.975) - 1.96) < 0.01
    assert abs(_inv_norm(0.025) + 1.96) < 0.01


def test_expected_max_z_grows_with_n():
    # Expected max of N standard normals strictly increases with N.
    e1 = _expected_max_z(1)
    e10 = _expected_max_z(10)
    e100 = _expected_max_z(100)
    e1000 = _expected_max_z(1000)
    assert e1 == 0.0
    assert e10 < e100 < e1000
    # Ballpark: expected max of 100 N(0,1) ≈ 2.5 (true value ≈ 2.508)
    assert 2.0 < e100 < 3.0


def test_deflated_sharpe_punishes_more_trials():
    # Same returns, but pretend we tried 1 vs 1000 strategies
    r = _normal_returns(mu=0.001, sigma=0.01, n=1000, seed=7)
    d1 = deflated_sharpe(r, 252, n_trials=1)
    d1000 = deflated_sharpe(r, 252, n_trials=1000)
    # DSR should drop with more trials (PSR bar gets higher)
    assert d1["dsr"] >= d1000["dsr"]
    # Threshold SR should increase with more trials
    assert d1000["sr_threshold"] > d1["sr_threshold"]


def test_deflated_sharpe_high_for_strong_signal_low_trials():
    # Strong mean, low noise, single trial → DSR should be very high
    r = _normal_returns(mu=0.002, sigma=0.005, n=2000, seed=11)
    d = deflated_sharpe(r, 252, n_trials=1)
    assert d["dsr"] > 0.95, f"strong single-trial signal failed DSR: {d['dsr']}"


def test_deflated_sharpe_low_for_weak_signal_high_trials():
    r = _normal_returns(mu=0.0001, sigma=0.01, n=500, seed=13)
    d = deflated_sharpe(r, 252, n_trials=10000)
    # Very weak signal + 10k trials → dsr should be low
    assert d["dsr"] < 0.5


def test_deflated_sharpe_short_series_returns_nan():
    r = _normal_returns(mu=0.001, sigma=0.01, n=10)
    d = deflated_sharpe(r, 252, n_trials=1)
    assert math.isnan(d["dsr"])
    assert d["n_obs"] == 10


# ---------- bars_per_year --------------------------------------------------

def test_bars_per_year_crypto_is_higher():
    assert bars_per_year("1h", "crypto") > bars_per_year("1h", "equity")
    assert bars_per_year("1d", "crypto") == 365
    assert bars_per_year("1d", "equity") == 252


# ---------- compute_stats integration --------------------------------------

def test_compute_stats_full_bundle():
    r = _normal_returns(mu=0.001, sigma=0.01, n=500, seed=17)
    eq = (1 + r).cumprod() * 100
    out = compute_stats(r, eq, interval="1d", asset_class="equity", n_trials=20)
    # Top-level keys present
    for k in ("sharpe", "sortino", "calmar", "cagr", "max_dd",
              "hit_rate", "deflated"):
        assert k in out, f"missing key: {k}"
    # Deflated nested correctly
    for k in ("sharpe", "psr", "dsr", "sr_threshold", "n_trials"):
        assert k in out["deflated"]
    assert out["deflated"]["n_trials"] == 20


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
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
