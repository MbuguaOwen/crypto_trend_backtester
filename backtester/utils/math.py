
import numpy as np
import pandas as pd

def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(b != 0, a / b, 0.0)
    return out

def rolling_donchian(series: pd.Series, lookback: int):
    highs = series.rolling(lookback, min_periods=lookback).max()
    lows  = series.rolling(lookback, min_periods=lookback).min()
    return highs, lows
