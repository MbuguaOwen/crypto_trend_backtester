# core/signals.py
import pandas as pd
import numpy as np
from .indicators import rolling_percentile, body_ratio

def compute_signals(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Very simple breakout-after-compression with confirmation.
    """
    out = bars.copy()
    cl = out["close"]
    atr = out["atr"].replace(0, np.nan)

    # Compression via rolling stdev of returns
    ret = cl.pct_change().fillna(0.0)
    look = cfg["signal"].get("compression_lookback", 80)
    pct  = cfg["signal"].get("compression_percentile", 70)
    stdev = ret.rolling(look, min_periods=max(10, look//2)).std()
    comp_threshold = rolling_percentile(stdev, look, pct)
    out["is_compressed"] = (stdev <= comp_threshold).astype(int)

    # Breakout level
    blb = cfg["windows"].get("breakout_lookback", 25)
    out["hh"] = out["high"].rolling(blb).max()
    out["ll"] = out["low"].rolling(blb).min()

    offset_atr = cfg["signal"].get("breakout_offset_atr", 0.75)
    confirm_n  = cfg["signal"].get("need_confirm_closes", 1)
    min_body   = cfg["signal"].get("min_body_ratio", 0.6)

    # confirmation signal
    out["body_ratio"] = body_ratio(out["open"], out["high"], out["low"], out["close"])
    long_breakout_line  = out["hh"] + offset_atr * atr
    short_breakout_line = out["ll"] - offset_atr * atr

    # signal flags on CLOSE (no lookahead)
    out["long_raw"]  = (out["is_compressed"].eq(1)) & (out["close"] > long_breakout_line)
    out["short_raw"] = (out["is_compressed"].eq(1)) & (out["close"] < short_breakout_line)

    # apply confirmation (need N consecutive bars with body quality)
    out["long_ok"]  = out["long_raw"] & (out["body_ratio"] >= min_body)
    out["short_ok"] = out["short_raw"] & (out["body_ratio"] >= min_body)

    if confirm_n > 1:
        out["long_signal"]  = out["long_ok"].rolling(confirm_n).apply(lambda x: int(x.all()), raw=False).fillna(0).astype(bool)
        out["short_signal"] = out["short_ok"].rolling(confirm_n).apply(lambda x: int(x.all()), raw=False).fillna(0).astype(bool)
    else:
        out["long_signal"]  = out["long_ok"]
        out["short_signal"] = out["short_ok"]

    return out
