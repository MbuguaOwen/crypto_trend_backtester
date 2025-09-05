from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from .features import atr, donchian, rolling_tstat_of_mean


def pass_micro_alignment(row: pd.Series, cfg_micro: Dict) -> bool:
    """
    Micro alignment gate:
      - abs(kappa_micro_short) >= tmin
      - acceleration = kappa_micro_short - kappa_micro_long >= accel_min
    """
    tshort = row.get("kappa_micro_short")
    tlong = row.get("kappa_micro_long")
    if pd.isna(tshort) or pd.isna(tlong):
        return False
    accel = tshort - tlong
    return (abs(tshort) >= float(cfg_micro["tmin"])) and (accel >= float(cfg_micro["accel_min"]))


def build_signal_frame(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Add ATR, Donchian, micro t-stats, acceleration, and raw breakout columns.
    Does NOT apply regime/micro gates—use backtest loop for gating + cooldown.
    """
    out = df.copy()
    aw = int(cfg["features"]["atr_window"])
    lookback = int(cfg["features"]["donchian_lookback"])
    short_w = int(cfg["micro"]["short_window"])
    long_w = int(cfg["micro"]["long_window"])
    buffer_mult = float(cfg["entry"]["atr_buffer_mult"])

    out["atr"] = atr(out, aw)
    d_hi, d_lo = donchian(out, lookback)
    out["d_hi"] = d_hi
    out["d_lo"] = d_lo

    # Micro t-stats on returns
    # Use log returns of close
    r1m = np.log(out["close"]).diff()
    out["kappa_micro_short"] = rolling_tstat_of_mean(r1m, short_w)
    out["kappa_micro_long"] = rolling_tstat_of_mean(r1m, long_w)
    out["accel"] = out["kappa_micro_short"] - out["kappa_micro_long"]

    # Raw ignition (structure) checks (without regime/micro gates)
    out["long_breakout"] = out["close"] > (out["d_hi"] + buffer_mult * out["atr"])
    out["short_breakout"] = out["close"] < (out["d_lo"] - buffer_mult * out["atr"])
    return out
