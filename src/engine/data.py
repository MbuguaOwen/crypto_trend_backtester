import os
import pandas as pd


def _parquet_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return base + ".parquet"


def load_symbol_1m(inputs_dir: str, symbol: str, months: list, progress: bool = False) -> pd.DataFrame:
    """
    Load 1m OHLCV across requested months.
    Uses a side-by-side .parquet cache per CSV (built once, then reused).
    Expected CSV columns: timestamp, open, high, low, close, volume
    """
    dfs = []
    for m in months:
        csv_path = os.path.join(inputs_dir, f"{symbol}-1m-{m}.csv")  # adapt to your naming if different
        pq_path = _parquet_path(csv_path)
        if os.path.exists(pq_path):
            df = pd.read_parquet(pq_path)
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(csv_path)
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values('timestamp').set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']].astype('float64')
            df.to_parquet(pq_path, compression='zstd', engine='pyarrow', index=True)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    out = pd.concat(dfs).sort_index()
    out.index = pd.DatetimeIndex(out.index, tz='UTC')
    return out

