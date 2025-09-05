from __future__ import annotations

import os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _parse_timestamp_col(ts: pd.Series) -> pd.DatetimeIndex:
    """
    Parse a 'timestamp' column that may be:
      - milliseconds since epoch (ints)
      - ISO-8601 strings
    Returns a UTC DatetimeIndex.
    """
    # If mostly numeric -> try ms since epoch.
    numeric = pd.to_numeric(ts, errors="coerce")
    numeric_non_na = numeric.notna().mean()
    if numeric_non_na > 0.5:
        # Heuristic: millisecond epoch is typically > 1e11
        if numeric.median(skipna=True) > 1e11:
            dt = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
        else:
            # seconds (fallback)
            dt = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ts, utc=True, errors="coerce")

    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Found {bad} unparsable timestamps; ensure 'timestamp' is ISO or epoch ms/s.")
    return pd.DatetimeIndex(dt)


def load_symbol_months(inputs_dir: str | Path, symbol: str, months: List[str]) -> pd.DataFrame:
    """
    Load 1m OHLCV CSVs for a symbol across a list of YYYY-MM months.
    Enforces UTC index, strictly increasing, drops dupes (keep last), and sorts.
    Does NOT infer missing bars.
    """
    inputs_dir = Path(inputs_dir)
    frames: List[pd.DataFrame] = []

    for ym in months:
        fpath = inputs_dir / symbol / f"{ym}.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing file: {fpath}")

        df = pd.read_csv(fpath)
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        missing = [c for c in REQUIRED_COLS if c not in cols]
        if missing:
            raise ValueError(f"{fpath} missing required columns: {missing}")

        # Parse timestamp to UTC index
        idx = _parse_timestamp_col(df["timestamp"])
        df = df.set_index(idx).drop(columns=["timestamp"])

        # Ensure numeric dtypes
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError(f"{fpath} contains non-numeric OHLC values.")

        df = df.sort_index()
        # Drop dupes (keep last)
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        after = len(df)
        # (optional) warn if duplicates dropped
        # print(f"{fpath}: dropped {before-after} duplicate timestamps")

        frames.append(df)

    if not frames:
        raise ValueError(f"No data loaded for {symbol}.")

    out = pd.concat(frames, axis=0).sort_index()
    if not out.index.is_monotonic_increasing:
        raise ValueError(f"Index not strictly increasing for {symbol}.")
    return out
