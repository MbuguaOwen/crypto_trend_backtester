from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def logret(close: pd.Series) -> pd.Series:
    """
    Compute 1-minute log returns: log(close).diff().
    """
    return np.log(close).diff()


def rolling_tstat_of_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling t-statistic of the mean over a fixed-size window.
    t = mean(x) / (std(x, ddof=1) / sqrt(n))
    NaN until the window is full or std == 0.
    """
    if window <= 1:
        raise ValueError("window must be >= 2 for a meaningful t-stat.")
    roll_mean = series.rolling(window=window, min_periods=window).mean()
    roll_std = series.rolling(window=window, min_periods=window).std(ddof=1)
    denom = (roll_std / np.sqrt(window))
    t = roll_mean / denom
    t = t.where(roll_std > 0)
    return t


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Average True Range using mean(TR) over a rolling window.
    TR = max(high - low, |high - prev_close|, |low - prev_close|).
    NaN until the window is full.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def donchian(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
    """
    Donchian channel using lookback highs/lows *excluding the current bar*.
    Returns (d_hi, d_lo) where both are shifted by 1 bar to avoid look-ahead.
    """
    d_hi = df["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    d_lo = df["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)
    return d_hi, d_lo
