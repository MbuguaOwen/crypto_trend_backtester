
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class StrategyParams:
    momentum_windows: List[int]
    breakout_lookback: int
    vol_window: int
    target_vol_annual: float
    max_leverage: float

def prepare_features(bars: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = bars.copy()
    df["logret"] = (df["close"]).pct_change().add(1).clip(lower=1e-12).pipe(lambda s: s.apply(lambda x: __import__("math").log(x)))
    df["rv_min"] = df["logret"].rolling(params.vol_window, min_periods=params.vol_window).std()
    df["donchian_hi"] = df["close"].rolling(params.breakout_lookback, min_periods=params.breakout_lookback).max()
    df["donchian_lo"] = df["close"].rolling(params.breakout_lookback, min_periods=params.breakout_lookback).min()
    moms = []
    for w in params.momentum_windows:
        col = f"mom_{w}"
        df[col] = df["close"].pct_change(w)
        moms.append(col)
    df["mom_score"] = df[moms].sum(axis=1).fillna(0.0)
    return df
