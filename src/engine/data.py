
import os
import pandas as pd
import numpy as np
from .utils import ensure_datetime_utc

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
    # handle ms/seconds/iso
    if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
        ts = ts.astype('int64')
        # guess ms vs s
        if ts.median() > 1e12:
            idx = pd.to_datetime(ts, unit='ms', utc=True)
        else:
            idx = pd.to_datetime(ts, unit='s', utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    df.index = idx.floor('T')
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
    frames = []
    for m in months:
        fn = f"{symbol}/{symbol}-ticks-{m}.csv"
        path = os.path.join(inputs_dir, fn)
        if not os.path.exists(path):
            continue
        if progress:
            print(f"[{symbol}] Loading {m} → {os.path.basename(path)}")
        frames.append(read_ticks_to_1m(path))
    if not frames:
        raise FileNotFoundError(f"No monthly files found for {symbol}. Looked for months={months}.")
    df = pd.concat(frames).sort_index()
    # dedup minutes if any overlap
    df = df[~df.index.duplicated(keep='last')]
    return df
