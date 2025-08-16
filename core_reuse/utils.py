from __future__ import annotations
import importlib
from typing import List
import pandas as pd

def try_import(paths: List[str], name: str):
    last_err = None
    for p in paths:
        try:
            return importlib.import_module(p)
        except Exception as e:
            last_err = e
    raise last_err if last_err else ImportError(f"Could not import {name}")

def rolling_atr(df: pd.DataFrame, window: int) -> pd.Series:
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def bb_width(close: pd.Series, window: int) -> pd.Series:
    ma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = ma + 2*std
    lower = ma - 2*std
    return (upper - lower) / ma

def pct_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: (x.rank(pct=True).iloc[-1] if len(x.dropna())==window else float("nan")), raw=False)
