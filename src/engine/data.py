import os
import pandas as pd
import numpy as np
from .utils import ensure_datetime_utc
from tqdm import tqdm


def _infer_timestamp_col(cols):
    for c in cols:
        lc = c.lower()
        if lc in ('timestamp','ts','time','datetime','date'):
            return c
    raise ValueError("No timestamp-like column found. Expected one of: timestamp, ts, time, datetime.")


def _infer_price_col(cols):
    for c in cols:
        lc = c.lower()
        if lc in ('price','p','last','close'):
            return c
    raise ValueError("No price-like column found. Expected one of: price, p, last, close.")


def _infer_qty_col(cols):
    for c in cols:
        lc = c.lower()
        if lc in ('qty','quantity','amount','size','volume','vol'):
            return c
    # volume not strictly required; use 1
    return None


def read_ticks_to_1m(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    ts_col = _infer_timestamp_col(df.columns)
    p_col  = _infer_price_col(df.columns)
    q_col  = _infer_qty_col(df.columns)
    ts = df[ts_col]
    # Always coerce to a DatetimeIndex, supporting mixed content robustly.
    # 1) If numeric-like → choose ms vs s by magnitude
    if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
        ts = pd.to_numeric(ts, errors='coerce')
        unit = 'ms' if float(ts.dropna().median()) > 1e12 else 's'
        dt = pd.to_datetime(ts, unit=unit, utc=True, errors='coerce')
    else:
        # string/mixed → strip quotes/whitespace then parse with format='mixed' if available
        s = ts.astype(str).str.strip().str.replace('"','', regex=False).replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        try:
            # pandas >= 2.0 supports format='mixed'
            dt = pd.to_datetime(s, utc=True, format='mixed', errors='coerce')
        except TypeError:
            # fallback: ISO8601 first, then general inference
            try:
                dt = pd.to_datetime(s, utc=True, format='ISO8601', errors='coerce')
            except Exception:
                dt = pd.to_datetime(s, utc=True, errors='coerce')
    # Drop rows that failed to parse
    bad = dt.isna()
    if bad.any():
        df = df.loc[~bad].copy()
        dt = dt[~bad]
    # use 'min' (not 'T') to avoid FutureWarning
    idx = pd.DatetimeIndex(dt).floor('min')
    df.index = idx
    price = df[p_col].astype('float64')
    vol = df[q_col].astype('float64') if q_col is not None else 1.0
    out = pd.DataFrame({
        'open': price.groupby(df.index).first(),
        'high': price.groupby(df.index).max(),
        'low' : price.groupby(df.index).min(),
        'close': price.groupby(df.index).last(),
        'volume': vol.groupby(df.index).sum()
    }).dropna()
    out.index.name = 'timestamp'
    return out


def load_symbol_1m(inputs_dir: str, symbol: str, months: list, progress=True):
    """Load 1-minute bars with optional Parquet caching."""
    cache_path = os.path.join(inputs_dir, f"{symbol}_1m.parquet")
    csv_paths = []
    for m in months:
        fn = f"{symbol}/{symbol}-ticks-{m}.csv"
        path = os.path.join(inputs_dir, fn)
        if os.path.exists(path):
            csv_paths.append(path)
    if csv_paths and os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        newest_csv = max(os.path.getmtime(p) for p in csv_paths)
        if cache_mtime >= newest_csv:
            return pd.read_parquet(cache_path)

    frames = []
    iterator = months
    bar = None
    if progress:
        bar = tqdm(months, desc=f"{symbol} months", ncols=100, leave=False)
        iterator = bar
    for m in iterator:
        fn = f"{symbol}/{symbol}-ticks-{m}.csv"
        path = os.path.join(inputs_dir, fn)
        if not os.path.exists(path):
            if not progress:
                print(f"[{symbol}] MISSING {m} → {os.path.basename(fn)}")
            continue
        if progress and bar is not None:
            bar.set_postfix_str(m)
        else:
            print(f"[{symbol}] Loading {m} → {os.path.basename(path)}")
        frames.append(read_ticks_to_1m(path))
    if bar is not None:
        bar.close()
    if not frames:
        raise FileNotFoundError(f"No monthly files found for {symbol}. Looked for months={months}.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    try:
        df.to_parquet(cache_path)
    except Exception:
        pass
    return df
