import os, glob, pandas as pd
from datetime import datetime, timezone
from tqdm.auto import tqdm

def _ensure_ts_index(df: pd.DataFrame):
    # Normalize columns to lowercase for robustness
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Candidate timestamp columns (most common first)
    candidates = ['timestamp', 'timestamp_ms', 'ts', 'time', 'date', 'datetime']
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError("No timestamp-like column found (expected one of: timestamp, timestamp_ms, ts, time, date, datetime)")

    s = df[col]

    # Detect numeric (int/float) or number-like strings
    def _is_numbery(x):
        if getattr(x, "dtype", None) is not None and x.dtype.kind in 'ifb':
            return True
        y = x.dropna().astype(str).str.strip()
        if y.empty:
            return False
        return y.str.fullmatch(r"[-+]?\d+(\.\d+)?").all()

    if _is_numbery(s):
        # epoch seconds or milliseconds; allow fractional seconds
        ser = pd.to_numeric(s, errors='coerce')
        med = ser.dropna().median()
        if pd.isna(med):
            raise ValueError(f"Timestamp column '{col}' could not be parsed as numeric.")
        if med > 1e11:  # clearly ms
            ts = pd.to_datetime(ser, unit='ms', utc=True)
        else:          # seconds (possibly fractional)
            ts = pd.to_datetime((ser * 1000).astype('int64'), unit='ms', utc=True)
    else:
        # ISO strings, possibly with microseconds and timezone
        s_str = s.astype(str).str.replace('Z', '+00:00', regex=False).str.strip()
        ts = pd.to_datetime(s_str, utc=True, errors='coerce')
        if ts.isna().any():
            # targeted fallbacks
            fmts = [
                "%Y-%m-%d %H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
            ]
            best = ts
            for fmt in fmts:
                try_ts = pd.to_datetime(s_str, format=fmt, utc=True, errors='coerce')
                if try_ts.notna().sum() > best.notna().sum():
                    best = try_ts
            ts = best

    if ts.isna().any():
        bad = int(ts.isna().sum())
        ex_idx = ts.isna().to_numpy().nonzero()[0]
        example = s.iloc[ex_idx[0]] if len(ex_idx) else s.iloc[0]
        raise ValueError(f"Failed to parse {bad} timestamps in column '{col}'. Example: {example}")

    df.index = ts
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df


def from_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_ts_index(df)
    cols = ['open','high','low','close','volume']
    if not set(cols).issubset(df.columns):
        raise ValueError(f"Missing columns in {path}; need {cols}")
    return df[cols].astype(float)

def from_tick_csv(path: str, chunksize: int | None = None, show_progress: bool = True) -> pd.DataFrame:
    # Chunked reader to get visible progress on huge tick CSVs.
    # Columns accepted:
    #   time:    timestamp | timestamp_ms | ts | time | date | datetime  (epoch s/ms or ISO)
    #   price:   price | p | last_price
    #   qty:     qty | size | q | amount | volume (optional)

    if chunksize is None or chunksize <= 0:
        # Simple path for small files
        df = pd.read_csv(path)
        df = _ensure_ts_index(df)
        price_col = next((c for c in ['price','p','last_price'] if c in df.columns), None)
        if price_col is None:
            raise ValueError(f"No price column found in {path}; expected one of: price, p, last_price")
        qty_col = next((c for c in ['qty','size','q','amount','volume'] if c in df.columns), None)

        p = df[price_col].astype(float)
        o = p.resample('1min').first()
        h = p.resample('1min').max()
        l = p.resample('1min').min()
        c = p.resample('1min').last()
        v = df[qty_col].astype(float).resample('1min').sum() if qty_col else None
        out = {'open': o, 'high': h, 'low': l, 'close': c}
        if v is not None: out['volume'] = v
        return pd.DataFrame(out).dropna(how='any')

    # Chunked path
    frames = []
    it = pd.read_csv(path, chunksize=chunksize)
    pbar = tqdm(it, desc=os.path.basename(path), unit="chunk", disable=(not show_progress))
    price_col, qty_col = None, None
    for chunk in pbar:
        chunk = _ensure_ts_index(chunk)
        if price_col is None:
            price_col = next((c for c in ['price','p','last_price'] if c in chunk.columns), None)
            if price_col is None:
                raise ValueError(f"No price column found in {path}; expected one of: price, p, last_price")
        if qty_col is None:
            qty_col = next((c for c in ['qty','size','q','amount','volume'] if c in chunk.columns), None)

        p = chunk[price_col].astype(float)
        o = p.resample('1min').first()
        h = p.resample('1min').max()
        l = p.resample('1min').min()
        c = p.resample('1min').last()
        if qty_col:
            v = chunk[qty_col].astype(float).resample('1min').sum()
        else:
            v = None
        out = {'open': o, 'high': h, 'low': l, 'close': c}
        if v is not None: out['volume'] = v
        frames.append(pd.DataFrame(out))
    if hasattr(pbar, "close"): pbar.close()

    if not frames:
        raise ValueError(f"Empty tick file: {path}")

    merged = pd.concat(frames, axis=0).sort_index()
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in merged.columns:
        agg['volume'] = 'sum'
    merged = merged.groupby(level=0).agg(agg)
    return merged.dropna(how='any')

def _expand_months(start_utc: str, end_utc: str):
    start = pd.Timestamp(start_utc, tz='UTC')
    end   = pd.Timestamp(end_utc, tz='UTC')
    months = pd.period_range(start, end - pd.Timedelta(days=1), freq='M')
    out = []
    for p in months:
        out.append({'yyyymm': f"{p.year}{p.month:02d}", 'yyyy': f"{p.year:04d}", 'mm': f"{p.month:02d}"})
    return out

def load_1m_df_for_range(symbol: str, cfg: dict) -> pd.DataFrame:
    bt = cfg['backtest']
    start = pd.Timestamp(bt['start_date_utc'], tz='UTC')
    end   = pd.Timestamp(bt['end_date_utc'], tz='UTC')

    typ = bt['inputs']['type']
    path_pattern = bt['inputs']['path_pattern']
    tick_pattern = bt['inputs'].get('tick_path_pattern', None)
    months = _expand_months(bt['start_date_utc'], bt['end_date_utc'])

    # Build candidate paths
    cand_paths = []
    if typ == 'ohlcv_csv':
        for m in months:
            cand_paths.append(path_pattern.format(symbol=symbol, **m))
    elif typ == 'tick_csv':
        for m in months:
            if tick_pattern:
                cand_paths.append(tick_pattern.format(symbol=symbol, **m))
            cand_paths.append(f"inputs/{symbol}/{symbol}-ticks-{m['yyyy']}-{m['mm']}.csv")
    else:
        raise ValueError(f"Unknown inputs.type: {typ}")

    file_paths = [p for p in cand_paths if os.path.exists(p)]
    frames = []

    if not file_paths:
        fallback = f"inputs/{symbol}/sample_1m.csv" if typ == 'ohlcv_csv' else f"inputs/{symbol}/sample_ticks.csv"
        if os.path.exists(fallback):
            file_paths = [fallback]

    if not file_paths:
        raise FileNotFoundError("No data files found for the requested range.")

    # Progress over files
    pbar = tqdm(total=len(file_paths), desc=f"{symbol}:load", unit="file", disable=(not cfg['backtest']['progress_bar']))
    try:
        for path in file_paths:
            try:
                if typ == 'ohlcv_csv':
                    frames.append(from_ohlcv_csv(path))
                else:
                    chunksize = bt['inputs'].get('tick_chunksize', None)
                    showp = cfg['backtest']['progress_bar']
                    frames.append(from_tick_csv(path, chunksize=chunksize, show_progress=showp))
            finally:
                pbar.update(1)
    finally:
        pbar.close()

    df = pd.concat(frames, axis=0).sort_index()
    df = df[(df.index >= start) & (df.index < end)]
    df = df[~df.index.duplicated(keep='last')]
    return df
