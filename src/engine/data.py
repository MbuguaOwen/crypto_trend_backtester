import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def _pick_ts_column(df: pd.DataFrame) -> str:
    """Find a timestamp column among common aliases."""
    for c in ('timestamp', 'ts', 'time'):
        if c in df.columns:
            return c
    raise KeyError("Expected a timestamp column named one of: 'timestamp', 'ts', 'time'.")


def _parse_ts(s: pd.Series) -> pd.DatetimeIndex:
    """
    Robustly parse tick timestamps that may be:
      - epoch milliseconds (int or numeric string, 13 digits)
      - epoch seconds (numeric)
      - ISO8601 strings with timezone (e.g., '2025-07-01 00:02:04+00:00' or '...Z')
    Returns tz-aware UTC DatetimeIndex, or raises a helpful error.
    """
    # Numeric fast-path (epoch ms vs s by magnitude)
    if np.issubdtype(s.dtype, np.number):
        s_int = s.astype("int64")
        unit = "s" if (s_int < 1_000_000_000_000).all() else "ms"
        return pd.to_datetime(s_int, unit=unit, utc=True)

    # Normalize strings
    s_str = s.astype(str).str.strip()
    # Normalize Z → +00:00 for consistency
    s_str = s_str.str.replace("Z", "+00:00", regex=False)
    # Treat empties / nans as invalid
    s_norm = s_str.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

    # 13-digit only? → epoch ms
    is_13 = s_norm.fillna("").str.match(r"^\d{13}$")
    if is_13.all():
        return pd.to_datetime(s_norm.astype("int64"), unit="ms", utc=True)

    # Try strict ISO8601 first (fast path)
    dt = pd.to_datetime(s_norm, utc=True, errors="coerce", format="ISO8601")

    # If some failed, try epoch seconds fallback for numeric-like strings
    if dt.isna().any():
        num = pd.to_numeric(s_norm, errors="coerce")
        dt_sec = pd.to_datetime(num, unit="s", utc=True, errors="coerce")
        # Prefer whichever parsed more rows
        if dt_sec.notna().sum() > dt.notna().sum():
            dt = dt_sec

    # Still failing? show a few problematic samples
    if dt.isna().any():
        bad_samples = s_str[dt.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(
            "Unparseable timestamp values (first few): "
            + "; ".join(repr(x) for x in bad_samples)
            + ". Expected epoch ms/seconds or ISO8601 with timezone "
              "(e.g., '2025-07-01 00:00:00.049000+00:00', '...+00:00', or '...Z')."
        )

    return pd.DatetimeIndex(dt)


def ticks_to_1m(df_ticks: pd.DataFrame) -> pd.DataFrame:
    """
    df_ticks columns: ['timestamp'|'ts'|'time','price','qty','is_buyer_maker'].
    Returns tz-aware UTC 1m OHLCV with ['open','high','low','close','volume'].
    """
    df = df_ticks.copy()

    # Robust timestamp parsing
    ts_col = _pick_ts_column(df)
    df['ts'] = _parse_ts(df[ts_col])

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
        # Let the parser handle the timestamp type
        df_ticks = pd.read_csv(path)
        frames.append(ticks_to_1m(df_ticks))
    if bar is not None:
        bar.close()
    if not frames:
        raise FileNotFoundError(f"No monthly files found for {symbol}. Looked for months={months}.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df
