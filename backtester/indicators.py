# backtester/indicators.py
from __future__ import annotations
import numpy as np, pandas as pd

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    hl = (high - low).abs()
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()

def bb_width_percentile(close: pd.Series, n: int = 50, pct_lookback: int = 50) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = ma + 2*sd
    lower = ma - 2*sd
    width = (upper - lower) / ma.clip(lower=1e-12)
    out = width.rolling(pct_lookback).apply(lambda x: 100.0 * (x.rank().iloc[-1] / len(x)) if len(x.dropna())==len(x) else np.nan, raw=False)
    return out

def ksigma_levels(close: pd.Series, n: int = 50):
    mu = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0).replace(0, np.nan)
    return {"mu": mu, "sd": sd}

def donchian_channels(high: pd.Series, low: pd.Series, n: int = 20):
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    return {"upper": upper, "lower": lower}

def realized_vol_ewma(returns: pd.Series, lam: float = 0.94, bar_minutes: int = 1) -> pd.Series:
    v = returns.pow(2).ewm(alpha=(1-lam), adjust=False).mean()
    bars_per_year = int(365*24*60 / max(1, bar_minutes))
    return np.sqrt(v * bars_per_year)

def tsmom_signal(close: pd.Series, lookbacks, consensus_min_abs: int = 1) -> pd.Series:
    sigs = []
    for L in lookbacks:
        retL = close.pct_change(L)
        sigs.append(np.sign(retL).fillna(0.0))
    agg = sum(sigs)
    out = agg.apply(lambda x: np.sign(x) if abs(x) >= consensus_min_abs else 0.0)
    return out.astype(int)
