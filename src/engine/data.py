import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def ticks_to_1m(df_ticks: pd.DataFrame) -> pd.DataFrame:
    """
    df_ticks columns: ['timestamp','price','qty','is_buyer_maker']
    timestamp in *milliseconds* since epoch.
    Returns tz-aware UTC 1m OHLCV with ['open','high','low','close','volume'].
    """
    df = df_ticks.copy()
    # ms -> UTC timestamp
    df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('ts').sort_index()

    # price*qty as turnover proxy; here 'volume' = sum(qty); you can also store notional
    ohlc = df['price'].resample('1min', label='right', closed='right').ohlc()
    vol  = df['qty'].resample('1min', label='right', closed='right').sum().rename('volume')
    out = pd.concat([ohlc, vol], axis=1).dropna()
    out.index = pd.DatetimeIndex(out.index, tz='UTC')
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
        df_ticks = pd.read_csv(path)
        frames.append(ticks_to_1m(df_ticks))
    if bar is not None:
        bar.close()
    if not frames:
        raise FileNotFoundError(f"No monthly files found for {symbol}. Looked for months={months}.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df
