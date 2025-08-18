# core/indicators.py
import pandas as pd
import numpy as np

def rolling_percentile(x: pd.Series, lookback: int, p: float) -> pd.Series:
    def _pct(window):
        if len(window) == 0:
            return np.nan
        return np.nanpercentile(window, p)
    return x.rolling(lookback, min_periods=lookback//2).apply(_pct, raw=False)

def body_ratio(o,h,l,c):
    rng = (h-l).replace(0, np.nan)
    body = (c-o).abs()
    return (body / rng).fillna(0.0)
