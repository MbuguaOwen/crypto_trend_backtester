# engine/signals.py
import numpy as np
import pandas as pd


def tstat(x: pd.Series) -> float:
    v = x.values
    mu = v.mean()
    sd = v.std(ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else mu / (sd / np.sqrt(len(v)))


def _percent_rank(a: np.ndarray) -> float:
    # Percentile rank of the last value within the window [0..100]
    if len(a) == 0 or np.all(np.isnan(a)):
        return np.nan
    x = a[~np.isnan(a)]
    if len(x) == 0:
        return np.nan
    last = x[-1]
    return 100.0 * (np.sum(x <= last) / len(x))


def compute_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    prev_close = close.shift(1)

    # ATR (classic TR mean)
    tr = pd.concat([
        (high - low).rename("hl"),
        (high - prev_close).abs().rename("hc"),
        (low - prev_close).abs().rename("lc"),
    ], axis=1).max(axis=1)
    atr_win = cfg["features"]["atr_window"]
    df["atr"] = tr.rolling(atr_win).mean()

    # Log-return t-stats
    lr = np.log(close).diff()
    mwin = cfg["regime"]["macro_window"]
    df["kappa"] = lr.rolling(mwin).apply(lambda s: tstat(s.fillna(0)), raw=False)

    s, L = cfg["micro"]["short_window"], cfg["micro"]["long_window"]
    df["t_short"] = lr.rolling(s).apply(lambda s_: tstat(s_.fillna(0)), raw=False)
    df["t_long"] = lr.rolling(L).apply(lambda s_: tstat(s_.fillna(0)), raw=False)
    df["accel"] = df["t_short"] - df["t_long"]

    # Donchian (exclude current bar)
    n = cfg["features"]["donchian_lookback"]
    df["don_hi"] = high.shift(1).rolling(n).max()
    df["don_lo"] = low.shift(1).rolling(n).min()

    # Candle body and energy measures
    body = (close - open_).abs()
    df["body_atr_mult"] = body / df["atr"]

    # ATR as % of price + rolling percentile (energy floor)
    atr_pct = (df["atr"] / close).clip(lower=0)
    win_p = int(cfg["filters"]["atr_pctile_window"])
    df["atr_pctile"] = atr_pct.rolling(win_p).apply(_percent_rank, raw=True)

    return df


def entry_signal(row_prev, cfg):
    # Regime
    if not np.isfinite(row_prev.kappa) or abs(row_prev.kappa) < cfg["regime"]["macro_tmin"]:
        return None
    dir_long = row_prev.kappa > 0

    # Micro alignment
    if not (
        np.isfinite(row_prev.t_short)
        and np.isfinite(row_prev.t_long)
        and np.isfinite(row_prev.accel)
    ):
        return None
    if (
        abs(row_prev.t_short) < cfg["micro"]["tmin"]
        or row_prev.accel < cfg["micro"]["accel_min"]
    ):
        return None

    # Energy filter
    if not np.isfinite(row_prev.atr_pctile) or row_prev.atr_pctile < cfg["filters"]["atr_pctile_min"]:
        return None

    # Momentum candle at breakout
    if (
        not np.isfinite(row_prev.body_atr_mult)
        or row_prev.body_atr_mult < cfg["entry"]["min_body_atr_mult"]
    ):
        return None

    buf = (
        cfg["entry"]["atr_buffer_mult"] * row_prev.atr if np.isfinite(row_prev.atr) else np.nan
    )
    if not np.isfinite(buf):
        return None

    # Donchian breakout in regime direction
    if dir_long and row_prev.close > (row_prev.don_hi + buf):
        return "long"
    if (not dir_long) and row_prev.close < (row_prev.don_lo - buf):
        return "short"
    return None

