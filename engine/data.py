# engine/data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

BINANCE_12 = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base","taker_buy_quote","ignore"
]
MINIMAL_6 = ["timestamp","open","high","low","close","volume"]

def _read_csv_try_header(path: Path) -> pd.DataFrame:
    """Try reading with header row; if it looks wrong, re-read headerless and assign names."""
    df0 = pd.read_csv(path)
    cols = [c.strip().lower() for c in df0.columns]
    df0.columns = cols

    has_full = set(BINANCE_12).issubset(cols)
    has_min  = set(MINIMAL_6).issubset(cols)

    # If columns are numeric-looking (headerless file), force header=None
    def _looks_headerless(_cols: list[str]) -> bool:
        try:
            # many headerless Binance dumps become numeric strings like "1743..."
            numericish = sum(c.replace(".", "", 1).isdigit() for c in _cols)
            return numericish >= max(3, len(_cols)//2)
        except Exception:
            return False

    if has_full or has_min:
        return df0

    if _looks_headerless(cols) or df0.shape[1] in (6, 12):
        # Re-read WITHOUT header and assign names
        df = pd.read_csv(path, header=None)
        n = df.shape[1]
        if n >= 12:
            df = df.iloc[:, :12]
            df.columns = BINANCE_12
        elif n >= 6:
            df = df.iloc[:, :6]
            df.columns = MINIMAL_6
        else:
            raise ValueError(f"{path} has {n} columns; expected >=6.")
        return df

    # Fallback: try header=None anyway
    df = pd.read_csv(path, header=None)
    n = df.shape[1]
    if n >= 12:
        df = df.iloc[:, :12]
        df.columns = BINANCE_12
    elif n >= 6:
        df = df.iloc[:, :6]
        df.columns = MINIMAL_6
    else:
        raise ValueError(f"{path} has unsupported columns: {cols} and width={n}")
    return df

def _parse_epoch_to_ms(series: pd.Series) -> np.ndarray:
    """Return epoch in **milliseconds** as int64 from s/ms/us inputs."""
    v = pd.to_numeric(series, errors="coerce")
    med = v.dropna().median()
    if med > 1e14:      # microseconds
        dt = pd.to_datetime(v, unit="us", utc=True, errors="coerce")
    elif med > 1e11:    # milliseconds
        dt = pd.to_datetime(v, unit="ms", utc=True, errors="coerce")
    else:               # seconds
        dt = pd.to_datetime(v, unit="s",  utc=True, errors="coerce")
    return (dt.view("int64") // 1_000_000).astype(np.int64)

def _standardize(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols

    if set(BINANCE_12).issubset(cols):
        out = pd.DataFrame({
            "timestamp": _parse_epoch_to_ms(df["open_time"]),
            "open":  pd.to_numeric(df["open"], errors="coerce"),
            "high":  pd.to_numeric(df["high"], errors="coerce"),
            "low":   pd.to_numeric(df["low"],  errors="coerce"),
            "close": pd.to_numeric(df["close"],errors="coerce"),
            "volume":pd.to_numeric(df["volume"],errors="coerce").fillna(0.0),
        })
    elif set(MINIMAL_6).issubset(cols):
        out = pd.DataFrame({
            "timestamp": _parse_epoch_to_ms(df["timestamp"]),
            "open":  pd.to_numeric(df["open"], errors="coerce"),
            "high":  pd.to_numeric(df["high"], errors="coerce"),
            "low":   pd.to_numeric(df["low"],  errors="coerce"),
            "close": pd.to_numeric(df["close"],errors="coerce"),
            "volume":pd.to_numeric(df["volume"],errors="coerce").fillna(0.0),
        })
    else:
        raise ValueError(f"{src} has unsupported columns: {cols}")

    out = out.dropna(subset=["open","high","low","close"]).astype({
        "timestamp":"int64","open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"
    })
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out

def _read_csv_standardize(path: Path) -> pd.DataFrame:
    df = _read_csv_try_header(path)
    return _standardize(df, path)

def load_symbol_months(inputs_dir: str, symbol: str, months: list[str]) -> pd.DataFrame:
    """
    Load multiple months for a symbol. Accept filenames:
      - inputs/<SYMBOL>/<YYYY-MM>.csv
      - inputs/<SYMBOL>/<SYMBOL>-1m-<YYYY-MM>.csv   (Binance)
    """
    base = Path(inputs_dir) / symbol
    parts = []
    for ym in months:
        p1 = base / f"{ym}.csv"
        p2 = base / f"{symbol}-1m-{ym}.csv"
        path = p1 if p1.exists() else p2
        if not path.exists():
            raise FileNotFoundError(f"Missing data for {symbol} {ym}: {p1} or {p2}")
        parts.append(_read_csv_standardize(path))
    if not parts:
        raise ValueError(f"No data for {symbol}")
    return pd.concat(parts, axis=0, ignore_index=True)
