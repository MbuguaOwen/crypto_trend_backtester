
import numpy as np
import pandas as pd

def _normalize_rule(rule: str) -> str:
    """Normalize pandas offset aliases to avoid FutureWarnings."""
    r = str(rule)
    r = r.replace('T', 'min')  # '5T' → '5min'
    r = r.replace('H', 'h')    # '1H' → '1h'
    return r

def ensure_datetime_utc(s):
    s = pd.to_datetime(s, utc=True)
    return s

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def true_range_last(row, prev_close):
    return max(row['high'] - row['low'],
               abs(row['high'] - prev_close),
               abs(row['low'] - prev_close))

def body_dom(row):
    rng = max(1e-9, row['high'] - row['low'])
    return abs((row['close'] - row['open']) / rng)

def zscore_logret(close: pd.Series, win: int = 20) -> pd.Series:
    lr = np.log(close / close.shift(1)).fillna(0.0)
    mu = lr.rolling(win, min_periods=1).mean()
    sd = lr.rolling(win, min_periods=1).std(ddof=0).replace(0, np.nan)
    z = (lr - mu) / sd
    return z.fillna(0.0)

def resample_ohlcv(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    rule = _normalize_rule(rule)
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return df1m.resample(rule).apply(agg).dropna()

def donchian_high(df: pd.DataFrame, lookback: int) -> pd.Series:
    return df['high'].rolling(lookback, min_periods=1).max()

def donchian_low(df: pd.DataFrame, lookback: int) -> pd.Series:
    return df['low'].rolling(lookback, min_periods=1).min()
