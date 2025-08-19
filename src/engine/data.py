
import os
import pandas as pd
from .utils import ensure_datetime_utc
import numpy as np
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
    # handle ms/seconds/iso → ALWAYS wrap into DatetimeIndex then floor
    if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
        ts = ts.astype('int64')
        unit = 'ms' if float(ts.median()) > 1e12 else 's'
        dt = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True)
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
    # dedup minutes if any overlap
    df = df[~df.index.duplicated(keep='last')]
    return df
