"""
Backtest performance statistics.

The agent will run parameter sweeps. Naive Sharpe ratios from sweeps overstate
alpha — the more variants you test, the more likely you are to find one with
a high Sharpe by chance. Deflated Sharpe (Bailey & Lopez de Prado, 2014)
corrects for this. Without it, the agent will confidently surface false
positives. With it, it won't. This is non-negotiable.

All functions take a returns Series indexed by ts_utc. The router and backtest
modules emit returns in this shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


# ---------- Annualization helpers -------------------------------------------

# bars per year, conservative for round numbers
_BARS_PER_YEAR = {
    "1m": 252 * 6.5 * 60,    # ~98k for equities; for crypto we override
    "5m": 252 * 6.5 * 12,
    "15m": 252 * 6.5 * 4,
    "1h": 252 * 6.5,
    "1d": 252,
}
_BARS_PER_YEAR_CRYPTO = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "1d": 365,
}


def bars_per_year(interval: str, asset_class: str = "equity") -> float:
    table = _BARS_PER_YEAR_CRYPTO if asset_class == "crypto" else _BARS_PER_YEAR
    if interval not in table:
        raise ValueError(f"unknown interval {interval!r}")
    return float(table[interval])


# ---------- Core stats ------------------------------------------------------

def sharpe(returns: pd.Series, periods_per_year: float,
           rf: float = 0.0) -> float:
    """Annualized Sharpe ratio. rf is annualized risk-free rate."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = r - rf_per
    sd = excess.std(ddof=1)
    if not np.isfinite(sd):
        return float("nan")
    # Guard against numerical noise: a "constant" series can have std ~ 1e-19
    # from accumulator drift. Compare against scale of the mean.
    scale = max(abs(excess.mean()), 1e-12)
    if sd <= 1e-12 * scale or sd == 0:
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float,
            rf: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    rf_per = rf / periods_per_year
    excess = r - rf_per
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    dd = downside.std(ddof=1)
    if not np.isfinite(dd) or dd == 0:
        return float("nan")
    scale = max(abs(excess.mean()), 1e-12)
    if dd <= 1e-12 * scale:
        return float("nan")
    return float(excess.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Max drawdown as a fraction (negative number) and its peak/trough timestamps."""
    e = equity.dropna()
    if len(e) < 2:
        return 0.0, None, None
    running_max = e.cummax()
    dd = (e - running_max) / running_max
    trough_idx = dd.idxmin()
    if pd.isna(trough_idx):
        return 0.0, None, None
    peak_idx = e.loc[:trough_idx].idxmax()
    return float(dd.min()), peak_idx, trough_idx


def calmar(returns: pd.Series, equity: pd.Series,
           periods_per_year: float) -> float:
    """Annualized return / |max drawdown|."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    cagr_ = cagr(equity, periods_per_year)
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return float("nan")
    return float(cagr_ / abs(mdd))


def cagr(equity: pd.Series, periods_per_year: float) -> float:
    e = equity.dropna()
    if len(e) < 2 or e.iloc[0] <= 0:
        return float("nan")
    n_periods = len(e) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")
    return float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1)


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    nonzero = r[r != 0]
    if len(nonzero) == 0:
        return float("nan")
    return float((nonzero > 0).mean())


def turnover(positions: pd.Series) -> float:
    """Average per-bar absolute change in position. Useful as a sanity check."""
    p = positions.dropna()
    if len(p) < 2:
        return 0.0
    return float(p.diff().abs().mean())


# ---------- Deflated Sharpe -------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected_max_z(n_trials: int) -> float:
    """Expected maximum of N i.i.d. standard normals.

    Closed-form approximation from Bailey & Lopez de Prado (2014):
        E[max_N] ≈ (1 - γ) Z(1 - 1/N) + γ Z(1 - 1/(N e))
    where γ is Euler-Mascheroni and Z is the inverse standard normal CDF.
    """
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329
    # inverse normal via scipy if available, else a simple Beasley-Springer-Moro
    try:
        from scipy.stats import norm  # type: ignore
        z1 = float(norm.ppf(1 - 1.0 / n_trials))
        z2 = float(norm.ppf(1 - 1.0 / (n_trials * math.e)))
    except ImportError:
        z1 = _inv_norm(1 - 1.0 / n_trials)
        z2 = _inv_norm(1 - 1.0 / (n_trials * math.e))
    return (1 - gamma) * z1 + gamma * z2


def _inv_norm(p: float) -> float:
    """Beasley-Springer-Moro inverse normal CDF, no SciPy needed."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def deflated_sharpe(
    returns: pd.Series,
    periods_per_year: float,
    n_trials: int = 1,
    sharpe_variance_across_trials: Optional[float] = None,
) -> dict:
    """Deflated Sharpe Ratio (DSR).

    Returns a dict with:
        sharpe        : annualized SR of the strategy
        n_trials      : number of strategies/configs tested
        sr_threshold  : the SR you'd expect from the best of N random strategies
        psr           : Probabilistic Sharpe Ratio (P(true SR > 0))
        dsr           : Deflated Sharpe Ratio (P(true SR > sr_threshold))
        skew, kurt    : higher moments used in the correction
        n_obs         : number of return observations

    Interpretation:
        dsr > 0.95: the result survives multiple-testing correction
        dsr < 0.95: the result is plausibly luck given how many configs were tried

    If you only ran one config, n_trials=1 reduces this to PSR.
    """
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return {
            "sharpe": float("nan"), "n_trials": n_trials,
            "sr_threshold": float("nan"), "psr": float("nan"),
            "dsr": float("nan"), "skew": float("nan"),
            "kurt": float("nan"), "n_obs": n,
            "notes": "fewer than 30 observations; DSR not meaningful",
        }
    sr_per = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    sr_ann = sr_per * math.sqrt(periods_per_year)

    # higher moments
    g3 = float(r.skew())
    g4 = float(r.kurt())  # pandas excess kurtosis (subtracts 3)

    # Standard error of SR per period (Mertens 2002):
    #   SE(SR) = sqrt( (1 - g3*SR + (g4)/4 * SR^2) / (n-1) )
    se_sr_per = math.sqrt(max(1e-12,
        (1 - g3 * sr_per + (g4 / 4.0) * sr_per ** 2) / (n - 1)
    ))

    # Threshold SR: expected max SR per period across n_trials random strategies
    # with cross-trial dispersion sigma_sr (per-period). If not supplied,
    # use SE(SR) as a reasonable default.
    sigma_sr = (sharpe_variance_across_trials ** 0.5
                if sharpe_variance_across_trials is not None
                else se_sr_per)
    sr_threshold_per = sigma_sr * _expected_max_z(n_trials)
    sr_threshold_ann = sr_threshold_per * math.sqrt(periods_per_year)

    # PSR: P(true SR > 0)
    psr = _norm_cdf((sr_per - 0.0) / se_sr_per)
    # DSR: P(true SR > sr_threshold)
    dsr = _norm_cdf((sr_per - sr_threshold_per) / se_sr_per)

    return {
        "sharpe": float(sr_ann),
        "n_trials": int(n_trials),
        "sr_threshold": float(sr_threshold_ann),
        "psr": float(psr),
        "dsr": float(dsr),
        "skew": g3,
        "kurt": g4,
        "n_obs": int(n),
    }


# ---------- Aggregator ------------------------------------------------------

@dataclass
class StatsBundle:
    sharpe: float
    sortino: float
    calmar: float
    cagr: float
    max_dd: float
    max_dd_peak: Optional[str]
    max_dd_trough: Optional[str]
    hit_rate: float
    n_obs: int
    interval: str
    asset_class: str
    deflated: dict


def compute_stats(
    returns: pd.Series,
    equity: pd.Series,
    interval: str,
    asset_class: str = "equity",
    n_trials: int = 1,
    rf: float = 0.0,
) -> dict:
    ppy = bars_per_year(interval, asset_class)
    mdd, peak, trough = max_drawdown(equity)
    bundle = StatsBundle(
        sharpe=sharpe(returns, ppy, rf),
        sortino=sortino(returns, ppy, rf),
        calmar=calmar(returns, equity, ppy),
        cagr=cagr(equity, ppy),
        max_dd=mdd,
        max_dd_peak=peak.isoformat() if peak is not None else None,
        max_dd_trough=trough.isoformat() if trough is not None else None,
        hit_rate=hit_rate(returns),
        n_obs=int(returns.dropna().shape[0]),
        interval=interval,
        asset_class=asset_class,
        deflated=deflated_sharpe(returns, ppy, n_trials=n_trials),
    )
    return asdict(bundle)
