from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window)
    def calc(y):
        if np.any(np.isnan(y)):
            return np.nan
        coef = np.polyfit(x, y, 1)
        return coef[0]
    return series.rolling(window).apply(calc, raw=True)


def macro_regime(minute_index: pd.Index, bars_1h: pd.DataFrame, bars_4h: pd.DataFrame,
                  bars_1d: pd.DataFrame, cfg: Dict) -> pd.Series:
    """Compute macro regime using multi‑horizon TSMOM consensus."""

    lookbacks = cfg.get("lookbacks", [20, 60, 120])
    score = pd.Series(0.0, index=minute_index)
    for tf_bars in [bars_1h, bars_4h, bars_1d]:
        lc = np.log(tf_bars["close"])
        for lb in lookbacks:
            r = lc.diff(lb)
            s = np.sign(r)
            s = s.reindex(minute_index, method="ffill").shift(1)
            score = score.add(s.fillna(0), fill_value=0)
    lc_daily = np.log(bars_1d["close"])
    slope = _rolling_slope(lc_daily, cfg.get("slope_window_days", 60))
    slope = np.sign(slope).reindex(minute_index, method="ffill").shift(1)
    score = score.add(slope.fillna(0), fill_value=0)
    bull_th = cfg.get("score_bull_threshold", 0)
    bear_th = cfg.get("score_bear_threshold", 0)
    regime = score.copy()
    regime[score > bull_th] = 1
    regime[score < -bear_th] = -1
    regime[(score <= bull_th) & (score >= -bear_th)] = 0
    return regime


def micro_slope(minute_index: pd.Index, bars_15m: pd.DataFrame, cfg: Dict) -> pd.Series:
    lc = np.log(bars_15m["close"])
    sl = _rolling_slope(lc, cfg.get("slope_window_bars", 48))
    sl = sl.reindex(minute_index, method="ffill").shift(1)
    return sl


def compression(minute_bars: pd.DataFrame, cfg: Dict) -> pd.Series:
    if not cfg.get("enabled", True):
        return pd.Series(True, index=minute_bars.index)
    lookback = cfg.get("lookback", 50)
    close = minute_bars["close"]
    ma = close.rolling(lookback).mean()
    sd = close.rolling(lookback).std()
    width = (2 * sd) / ma
    def rank_pct(s):
        if s.isna().any():
            return np.nan
        return percentileofscore(s[:-1], s.iloc[-1])
    pct = width.rolling(lookback).apply(rank_pct, raw=False)
    pct = pct.shift(1)
    return pct <= cfg.get("percentile", 35)


def donchian(minute_bars: pd.DataFrame, lookback: int) -> pd.DataFrame:
    high = minute_bars["high"].rolling(lookback).max().shift(1)
    low = minute_bars["low"].rolling(lookback).min().shift(1)
    return pd.DataFrame({"high": high, "low": low})
