# core/bar_builder.py
import pandas as pd
import numpy as np


def _infer_ts_series(df):
    for c in ["timestamp", "ts", "time", "T"]:
        if c in df.columns:
            return df[c]
    raise ValueError("Tick CSV must have a 'timestamp' column (one of: timestamp, ts, time, T).")


def _parse_timestamp_col(ts: pd.Series) -> pd.Series:
    """
    Robustly parse timestamp column that may be:
      - numeric epoch in milliseconds or seconds
      - ISO8601 strings, possibly with timezone (+00:00 / Z)
      - mixed string formats
    Always returns UTC-aware pandas datetime64[ns, UTC].
    """
    # Numeric? (epoch)
    if np.issubdtype(ts.dtype, np.number):
        x = pd.to_numeric(ts, errors="coerce")
        # Heuristic: ms if > 10^12, else seconds
        unit = "ms" if float(np.nanmax(x)) > 1e12 else "s"
        return pd.to_datetime(x, unit=unit, utc=True)

    # String-like: try fast/strict → mixed → inferred fallback
    try:
        # Pandas 2.x fast path for ISO8601
        return pd.to_datetime(ts, format="ISO8601", utc=True)
    except Exception:
        pass
    try:
        # Pandas 2.x mixed formats
        return pd.to_datetime(ts, format="mixed", utc=True)
    except Exception:
        pass

    # Final fallback with inference; validate NaT ratio
    dt = pd.to_datetime(ts, utc=True, errors="coerce", infer_datetime_format=True)
    nat_ratio = float(dt.isna().mean())
    if nat_ratio > 0.01:
        bad = ts[dt.isna()].head(5).tolist()
        raise ValueError(
            f"Failed to parse timestamps: {nat_ratio:.2%} NaT. "
            f"First problematic examples: {bad}"
        )
    return dt


def read_ticks_to_bars(path: str, bar_minutes: int = 1) -> pd.DataFrame:
    """
    Expect CSV with columns: timestamp (ms/s epoch or ISO8601), price, qty (qty optional).
    """
    df = pd.read_csv(path)

    ts = _infer_ts_series(df)
    dt = _parse_timestamp_col(ts)

    # Floor to bar interval (fixes FutureWarning by using 'min' instead of 'T')
    df["_dt"] = dt.dt.floor(f"{bar_minutes}min")

    # price column
    px_col = None
    for c in ["price", "px", "p"]:
        if c in df.columns:
            px_col = c
            break
    if px_col is None:
        raise ValueError("Tick CSV must have a 'price' column.")

    qty_col = "qty" if "qty" in df.columns else None

    grouped = df.groupby("_dt", sort=True)
    o = grouped[px_col].first()
    h = grouped[px_col].max()
    l = grouped[px_col].min()
    c = grouped[px_col].last()
    v = grouped[qty_col].sum() if qty_col else grouped.size().astype(float)

    bars = (
        pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
        .reset_index()
        .rename(columns={"_dt": "ts"})
        .sort_values("ts")
        .reset_index(drop=True)
    )

    # ATR (simple moving average of True Range)
    tr = np.maximum(
        bars["high"] - bars["low"],
        np.maximum(
            (bars["high"] - bars["close"].shift()).abs(),
            (bars["low"] - bars["close"].shift()).abs(),
        ),
    )
    bars["atr"] = tr.rolling(14, min_periods=1).mean()
    return bars
