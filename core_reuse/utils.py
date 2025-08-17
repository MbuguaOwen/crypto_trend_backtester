import math, time, numpy as np, pandas as pd

def bps_to_frac(bps: float) -> float:
    return float(bps) / 1_0000.0

def utc_ms_now() -> int:
    return int(time.time() * 1000)

def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std()

def ema(series: pd.Series, halflife: int) -> pd.Series:
    return series.ewm(halflife=halflife, adjust=False).mean()
