import os
import pandas as pd
import numpy as np

# ---------- Schema helpers ----------

_CANON_PRICE_NAMES = ["price", "px", "last", "trade_price"]
_CANON_QTY_NAMES   = ["quantity", "qty", "size", "amount", "volume"]  # amount/volume may be quote/base; we handle below
_CANON_TS_NAMES    = ["timestamp", "ts", "time", "date", "datetime"]

def _choose_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for name in candidates:
        if name in cols_l:
            return cols_l[name]
    return None

def _detect_time_unit(series):
    """Best-effort detection of epoch unit for integer timestamps."""
    s = pd.Series(series.dropna().values[:1000])  # sample
    if s.empty:
        return None
    vmax = s.astype("int64").max()
    if vmax > 10**17:   # nanoseconds
        return "ns"
    if vmax > 10**12:   # milliseconds
        return "ms"
    if vmax > 10**10:   # microseconds (rare)
        return "us"
    return "s"          # seconds by default

def _normalize_ticks_columns(df):
    """Return df with columns: timestamp, price, quantity (quantity optional)."""
    cols = list(df.columns)
    ts_col   = _choose_col(cols, _CANON_TS_NAMES)
    px_col   = _choose_col(cols, _CANON_PRICE_NAMES)
    qty_col  = _choose_col(cols, _CANON_QTY_NAMES)

    if ts_col is None or px_col is None:
        raise ValueError(f"Missing required columns. Found={cols}, need timestamp & price.")

    df = df.rename(columns={ts_col: "timestamp", px_col: "price"})
    if qty_col and qty_col not in ("amount", "volume"):
        df = df.rename(columns={qty_col: "quantity"})

    # If we only have "amount" and "price", derive quantity = amount / price
    if "quantity" not in df.columns:
        if "amount" in df.columns and "price" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                qty = df["amount"] / df["price"]
            df["quantity"] = qty.fillna(0.0).astype("float64")
        elif "volume" in df.columns:
            # Treat "volume" as quantity if present
            df["quantity"] = df["volume"].astype("float64")
        else:
            # Quantity truly missing -> set zeros; ATR logic doesn’t need volume
            df["quantity"] = 0.0

    # Parse timestamp robustly
    if np.issubdtype(df["timestamp"].dtype, np.number):
        unit = _detect_time_unit(df["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit=unit, utc=True)
    else:
        # ISO8601 or mixed strings
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
        # Fallback if some rows failed
        mask = df["timestamp"].isna()
        if mask.any():
            df.loc[mask, "timestamp"] = pd.to_datetime(df.loc[mask, "timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            df = df.dropna(subset=["timestamp"])

    df = df[["timestamp", "price", "quantity"]].sort_values("timestamp").reset_index(drop=True)
    return df

# ---------- CSV reading ----------

def read_ticks(path: str) -> pd.DataFrame:
    """
    Read tick CSV robustly:
      - Try engine='pyarrow' (no memory_map) then fall back to pandas C engine
      - Auto-map timestamp/price/quantity columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Try Arrow fast path
    try:
        df = pd.read_csv(path, engine="pyarrow")   # memory_map not supported by pyarrow
    except Exception:
        # Fallback to pandas engine (usecols omitted to avoid 'Usecols' mismatches)
        df = pd.read_csv(path)

    df = _normalize_ticks_columns(df)
    return df

# ---------- Resampling ----------

def ticks_to_ohlcv_1m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ticks → 1-minute OHLCV.
    Volume is sum of 'quantity' (zeros if missing upstream).
    """
    if df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume","atr"])

    df["_minute"] = df["timestamp"].dt.floor("T")
    gp = df.groupby("_minute", sort=True)

    o = gp["price"].first().rename("open")
    h = gp["price"].max().rename("high")
    l = gp["price"].min().rename("low")
    c = gp["price"].last().rename("close")
    v = gp["quantity"].sum().rename("volume")

    bars = pd.concat([o, h, l, c, v], axis=1).reset_index().rename(columns={"_minute": "time"})
    bars.set_index("time", inplace=True)

    # Vectorized ATR (classic TR rolling mean)
    tr1 = bars["high"] - bars["low"]
    tr2 = (bars["high"] - bars["close"].shift()).abs()
    tr3 = (bars["low"]  - bars["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use your configured window at runtime; default=14 if not present
    # (Caller can overwrite bars['atr'] later if needed.)
    bars["atr"] = tr.rolling(14, min_periods=1).mean()

    return bars

# ---------- Loader used by backtest ----------

def load_symbol_1m(inputs_dir: str, symbol: str, months: list[str], progress: bool=False) -> pd.DataFrame:
    """
    Load per-month tick CSVs, concatenate, resample to 1m OHLCV.
    Expected file names: {symbol}-ticks-YYYY-MM.csv under inputs_dir.
    """
    parts = []
    for m in months:
        fn = f"{symbol}-ticks-{m}.csv"
        path = os.path.join(inputs_dir, fn)
        if not os.path.exists(path):
            # Skip missing months; the backtest will just use what’s available
            continue
        ticks = read_ticks(path)
        bars = ticks_to_ohlcv_1m(ticks)
        parts.append(bars)

    if not parts:
        return pd.DataFrame(columns=["open","high","low","close","volume","atr"])

    df1m = pd.concat(parts, axis=0).sort_index()
    # Drop duplicate minutes if any overlap across files
    df1m = df1m[~df1m.index.duplicated(keep="last")]
    return df1m