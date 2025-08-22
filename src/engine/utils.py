
import numpy as np
import pandas as pd


def _normalize_rule(rule: str) -> str:
    """Normalize pandas offset aliases to avoid FutureWarnings."""
    r = str(rule)
    aliases = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "5T": "5min",
    }
    return aliases.get(r, r)

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
    sd = lr.rolling(win, min_periods=1).std(ddof=0)
    arr_lr, arr_mu, arr_sd = lr.to_numpy(), mu.to_numpy(), sd.to_numpy()
    z = np.zeros_like(arr_lr)
    np.divide(arr_lr - arr_mu, arr_sd, out=z, where=(arr_sd > 0))
    np.nan_to_num(z, copy=False)
    return pd.Series(z, index=close.index)

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


def atr_vec(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Vectorized ATR (EWMA of true range)."""
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return pd.Series(tr).ewm(alpha=1 / window, adjust=False).mean().to_numpy()


def zscore_logret_vec(close: np.ndarray, win: int) -> np.ndarray:
    prev = np.concatenate(([close[0]], close[:-1]))
    lr = np.log(close / prev)
    lr[0] = 0.0
    mu = pd.Series(lr).rolling(win, min_periods=1).mean().to_numpy()
    sd = pd.Series(lr).rolling(win, min_periods=1).std(ddof=0).to_numpy()
    z = np.zeros_like(lr)
    np.divide(lr - mu, sd, out=z, where=(sd > 0))
    np.nan_to_num(z, copy=False)
    return z


def body_dom_vec(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    rng = np.maximum(1e-9, high - low)
    return np.abs((close - open_) / rng)


def true_range_vec(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
