# engine/data.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np

KLINE_COLSETS = [
    # Binance export columns (common)
    ["open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"],
    # Minimal OHLCV
    ["timestamp","open","high","low","close","volume"],
]

def _read_csv_standardize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    if set(KLINE_COLSETS[0]).issubset(cols):  # Binance full
        ts = pd.to_numeric(df["open_time"], errors="coerce")
        if ts.median(skipna=True) > 1e11:
            dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        out = pd.DataFrame({
            "timestamp": (dt.view("int64") // 1_000_000).astype(np.int64),
            "open":  pd.to_numeric(df["open"], errors="coerce"),
            "high":  pd.to_numeric(df["high"], errors="coerce"),
            "low":   pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume":pd.to_numeric(df["volume"], errors="coerce").fillna(0.0),
        })
    elif set(KLINE_COLSETS[1]).issubset(cols):  # already minimal
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.median(skipna=True) > 1e11:
            dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        out = pd.DataFrame({
            "timestamp": (dt.view("int64") // 1_000_000).astype(np.int64),
            "open":  pd.to_numeric(df["open"], errors="coerce"),
            "high":  pd.to_numeric(df["high"], errors="coerce"),
            "low":   pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume":pd.to_numeric(df["volume"], errors="coerce").fillna(0.0),
        })
    else:
        raise ValueError(f"{path} has unsupported columns: {cols}")

    out = out.dropna(subset=["open","high","low","close"]).astype({
        "timestamp":"int64","open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"
    })
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp")
    return out.reset_index(drop=True)

def load_symbol_months(inputs_dir: str, symbol: str, months: list[str]) -> pd.DataFrame:
    """Load multiple months for a symbol. Accept filenames:
    - inputs/<SYMBOL>/<YYYY-MM>.csv
    - inputs/<SYMBOL>/<SYMBOL>-1m-<YYYY-MM>.csv
    """
    base = Path(inputs_dir) / symbol
    parts = []
    for ym in months:
        # Try both patterns
        p1 = base / f"{ym}.csv"
        p2 = base / f"{symbol}-1m-{ym}.csv"
        path = p1 if p1.exists() else p2
        if not path.exists():
            raise FileNotFoundError(f"Missing data for {symbol} {ym}: {p1} or {p2}")
        parts.append(_read_csv_standardize(path))
    if not parts:
        raise ValueError(f"No data for {symbol}")
    df = pd.concat(parts, axis=0, ignore_index=True)
    return df
