# engine/signals.py
import numpy as np
import pandas as pd

def tstat(x: pd.Series) -> float:
    v = x.values
    mu = v.mean()
    sd = v.std(ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else mu / (sd / np.sqrt(len(v)))

def compute_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    # ATR
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = (high - low).to_frame("hl")
    tr["hc"] = (high - prev_close).abs()
    tr["lc"] = (low - prev_close).abs()
    df["atr"] = tr.max(axis=1).rolling(cfg["features"]["atr_window"]).mean()

    # returns and t-stats
    lr = np.log(close).diff()
    mwin = cfg["regime"]["macro_window"]
    df["kappa"] = lr.rolling(mwin).apply(lambda s: tstat(s.fillna(0)), raw=False)

    s, L = cfg["micro"]["short_window"], cfg["micro"]["long_window"]
    df["t_short"] = lr.rolling(s).apply(lambda s_: tstat(s_.fillna(0)), raw=False)
    df["t_long"]  = lr.rolling(L).apply(lambda s_: tstat(s_.fillna(0)), raw=False)
    df["accel"]   = df["t_short"] - df["t_long"]

    # Donchian (exclude current bar)
    n = cfg["features"]["donchian_lookback"]
    df["don_hi"] = high.shift(1).rolling(n).max()
    df["don_lo"] = low.shift(1).rolling(n).min()

    return df

def entry_signal(row_prev, cfg):
    # Regime gating
    if not np.isfinite(row_prev.kappa) or abs(row_prev.kappa) < cfg["regime"]["macro_tmin"]:
        return None
    # Cooldown handled outside if you keep state; skip here for brevity.

    # Micro alignment
    if not (np.isfinite(row_prev.t_short) and np.isfinite(row_prev.t_long) and np.isfinite(row_prev.accel)):
        return None
    if abs(row_prev.t_short) < cfg["micro"]["tmin"] or row_prev.accel < cfg["micro"]["accel_min"]:
        return None

    # Direction via regime sign
    dir_long = row_prev.kappa > 0
    buf = cfg["entry"]["atr_buffer_mult"] * row_prev.atr if np.isfinite(row_prev.atr) else np.nan
    if not np.isfinite(buf):
        return None

    if dir_long and row_prev.close > (row_prev.don_hi + buf):
        return "long"
    if (not dir_long) and row_prev.close < (row_prev.don_lo - buf):
        return "short"
    return None
