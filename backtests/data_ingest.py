from __future__ import annotations
import pandas as pd

TICK_COLS = ["timestamp_ms", "price", "qty", "is_buyer_maker"]

def _to_utc_index(series):
    # Accept ms integer or ISO string
    try:
        # If integer-like ms
        if pd.api.types.is_integer_dtype(series) or str(series.iloc[0]).isdigit():
            return pd.to_datetime(series.astype('int64'), unit='ms', utc=True)
    except Exception:
        pass
    return pd.to_datetime(series, utc=True)

def from_tick_csv(path: str) -> pd.DataFrame:
    """Aggregate trades to strict 1-minute OHLCV with UTC DatetimeIndex."""
    df = pd.read_csv(path)
    missing = [c for c in TICK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing tick cols: {missing}")
    df = df.drop_duplicates(subset=["timestamp_ms","price","qty"], keep="last")
    df["ts"] = _to_utc_index(df["timestamp_ms"])
    df.set_index("ts", inplace=True)
    df = df.sort_index()

    # 1-minute resample
    o = df["price"].resample("1min").first()
    h = df["price"].resample("1min").max()
    l = df["price"].resample("1min").min()
    c = df["price"].resample("1min").last()
    v = df["qty"].resample("1min").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    out = out.dropna(how="any")
    out.index.name = "ts"
    return out

def from_ohlcv_csv(path: str) -> pd.DataFrame:
    """Parse native 1m OHLCV; ts may be ms or ISO; ensure UTC index."""
    df = pd.read_csv(path)
    # flexible cols
    if "ts" in df.columns:
        ts = df["ts"]
    elif "timestamp" in df.columns:
        ts = df["timestamp"]
    else:
        # Some files encode ms in 'time' or 'open_time'
        for cand in ["time","open_time"]:
            if cand in df.columns:
                ts = df[cand]
                break
        else:
            raise ValueError(f"{path} missing ts/timestamp/time/open_time")
    idx = _to_utc_index(ts)
    df.index = idx
    cols = {}
    for k in ["open","high","low","close"]:
        if k not in df.columns:
            raise ValueError(f"{path} missing {k}")
        cols[k]=df[k].astype(float)
    cols["volume"] = df.get("volume", 0.0)
    out = pd.DataFrame(cols).sort_index()
    out = out.dropna(how="any")
    out.index.name="ts"
    return out

def load_1m_df(path: str) -> pd.DataFrame:
    """Auto-detect source by header and return strict, monotonic 1m OHLCV."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline().lower()
    if "timestamp_ms" in head and "price" in head and "qty" in head:
        df = from_tick_csv(path)
    else:
        df = from_ohlcv_csv(path)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
