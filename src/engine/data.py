
import os
import pandas as pd
from tqdm import tqdm


def read_ticks(path):
    usecols = ["timestamp", "price", "quantity", "is_buyer_maker"]
    try:
        df = pd.read_csv(path, usecols=usecols, engine="pyarrow", memory_map=True)
    except Exception:
        df = pd.read_csv(path, usecols=usecols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, infer_datetime_format=True)
    df = df.sort_values('timestamp')
    return df


def ticks_to_ohlcv_1m(df):
    # floor to minute
    df['_minute'] = df['timestamp'].dt.floor('1min')
    gp = df.groupby('_minute', sort=True)

    o = gp['price'].first().rename("open")
    h = gp['price'].max().rename("high")
    l = gp['price'].min().rename("low")
    c = gp['price'].last().rename("close")
    v = gp['quantity'].sum().rename("volume")

    bars = pd.concat([o, h, l, c, v], axis=1).reset_index().rename(columns={'_minute': 'time'})
    bars.set_index('time', inplace=True)
    return bars

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
        ticks = read_ticks(path)
        frames.append(ticks_to_ohlcv_1m(ticks))
    if bar is not None:
        bar.close()
    if not frames:
        raise FileNotFoundError(f"No monthly files found for {symbol}. Looked for months={months}.")
    df = pd.concat(frames).sort_index()
    # dedup minutes if any overlap
    df = df[~df.index.duplicated(keep='last')]
    return df
