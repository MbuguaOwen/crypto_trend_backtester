from __future__ import annotations

import os
from typing import List

import pandas as pd


def load_ticks(symbol: str, months: List[str], data_dir: str) -> pd.DataFrame:
    """Load tick data for ``symbol`` from ``data_dir``.

    The contract expects CSV files named ``{symbol}-ticks-{YYYY-MM}.csv`` with at
    least the columns ``ts`` (milliseconds), ``price`` and ``qty``.
    """

    dfs = []
    for m in months:
        path = os.path.join(data_dir, f"{symbol}-ticks-{m}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cols = [c for c in ["ts", "price", "qty"] if c in df.columns]
        df = df[cols]
        df = df[(df["price"] > 0) & (df["qty"] > 0)]
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No tick files found for {symbol}")
    df = pd.concat(dfs).drop_duplicates(subset="ts").sort_values("ts")
    df = df[df["ts"].diff().fillna(1) > 0]  # enforce monotonic
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df[["price", "qty"]]


def build_ohlcv(ticks: pd.DataFrame) -> pd.DataFrame:
    """Create strictly causal 1m OHLCV bars from ticks."""

    price = ticks["price"]
    qty = ticks["qty"]
    ohlc = price.resample("1T", label="right", closed="right").ohlc()
    vol = qty.resample("1T", label="right", closed="right").sum()
    vwap = (price * qty).resample("1T", label="right", closed="right").sum() / vol
    bars = ohlc.join(vol.rename("volume")).join(vwap.rename("vwap"))
    bars = bars.dropna()
    return bars


def resample_from_1m(bars_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample ``bars_1m`` to a higher frequency ``rule`` using only 1m data."""

    df = pd.DataFrame()
    df["open"] = bars_1m["open"].resample(rule, label="right", closed="right").first()
    df["high"] = bars_1m["high"].resample(rule, label="right", closed="right").max()
    df["low"] = bars_1m["low"].resample(rule, label="right", closed="right").min()
    df["close"] = bars_1m["close"].resample(rule, label="right", closed="right").last()
    df["volume"] = bars_1m["volume"].resample(rule, label="right", closed="right").sum()
    num = (bars_1m["vwap"] * bars_1m["volume"]).resample(rule, label="right", closed="right").sum()
    df["vwap"] = num / df["volume"]
    return df.dropna()
