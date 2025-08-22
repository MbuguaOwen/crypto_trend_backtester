import os, glob
import pandas as pd
import numpy as np

# ---------- Schema helpers ----------

_CANON_PRICE_NAMES = ["price", "px", "last", "trade_price"]
_CANON_QTY_NAMES   = ["quantity", "qty", "size", "amount", "volume"]  # 'qty' per your sample
_CANON_TS_NAMES    = ["timestamp", "ts", "time", "date", "datetime"]

def _choose_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for name in candidates:
        if name in cols_l:
            return cols_l[name]
    return None

def _detect_time_unit(series):
    """Detect epoch unit for integer timestamps."""
    s = pd.Series(series.dropna().values[:1000])
    if s.empty:
        return None
    vmax = s.astype("int64").max()
    if vmax > 10**17: return "ns"
    if vmax > 10**12: return "ms"   # your sample
    if vmax > 10**10: return "us"
    return "s"

def _normalize_ticks_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with timestamp, price, quantity (quantity optional)."""
    cols = list(df.columns)
    ts_col  = _choose_col(cols, _CANON_TS_NAMES)
    px_col  = _choose_col(cols, _CANON_PRICE_NAMES)
    qty_col = _choose_col(cols, _CANON_QTY_NAMES)

    if ts_col is None or px_col is None:
        raise ValueError(f"Missing required columns. Found={cols}, need timestamp & price.")

    df = df.rename(columns={ts_col: "timestamp", px_col: "price"})
    if qty_col and qty_col not in ("amount", "volume"):
        df = df.rename(columns={qty_col: "quantity"})

    # Derive quantity if only amount/volume present
    if "quantity" not in df.columns:
        if "amount" in df.columns and "price" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                qty = df["amount"] / df["price"]
            df["quantity"] = qty.fillna(0.0).astype("float64")
        elif "volume" in df.columns:
            df["quantity"] = df["volume"].astype("float64")
        else:
            df["quantity"] = 0.0

    # Parse timestamp: epoch ints or ISO strings
    if np.issubdtype(df["timestamp"].dtype, np.number):
        unit = _detect_time_unit(df["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
        bad = df["timestamp"].isna()
        if bad.any():
            df.loc[bad, "timestamp"] = pd.to_datetime(df.loc[bad, "timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

    df = df[["timestamp", "price", "quantity"]].sort_values("timestamp").reset_index(drop=True)
    return df

# ---------- CSV reading ----------

def _read_csv_any(path: str) -> pd.DataFrame:
    # Try pyarrow first (fast), then fallback to pandas engine. Don't use memory_map for pyarrow.
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, low_memory=False)

def read_ticks(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = _read_csv_any(path)
    return _normalize_ticks_columns(df)

# ---------- Resampling ----------

def ticks_to_ohlcv_1m(df: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume","atr"])

    df["_minute"] = df["timestamp"].dt.floor("T")
    gp = df.groupby("_minute", sort=True)

    o = gp["price"].first().rename("open")
    h = gp["price"].max().rename("high")
    l = gp["price"].min().rename("low")
    c = gp["price"].last().rename("close")
    v = gp["quantity"].sum().rename("volume")

    bars = pd.concat([o,h,l,c,v], axis=1).reset_index().rename(columns={"_minute": "time"})
    bars.set_index("time", inplace=True)

    # Vectorized ATR (classic TR rolling mean)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["close"].shift()).abs(),
        (bars["low"]  - bars["close"].shift()).abs()
    ], axis=1).max(axis=1)
    bars["atr"] = tr.rolling(atr_window, min_periods=1).mean()

    return bars

# ---------- Loader with diagnostics ----------

def _candidate_files(inputs_dir: str, symbol: str, month: str) -> list[str]:
    exact = os.path.join(inputs_dir, f"{symbol}-ticks-{month}.csv")
    if os.path.exists(exact):
        return [exact]
    # Glob fallback (case-insensitive-ish)
    patt = os.path.join(inputs_dir, f"*{symbol}*{month}*.csv")
    return sorted(glob.glob(patt))

def load_symbol_1m(inputs_dir: str, symbol: str, months: list[str], atr_window: int, diagnostics: dict) -> pd.DataFrame:
    """
    Load per-month tick CSVs, concatenate, resample to 1m OHLCV.
    Fills `diagnostics[symbol]` with per-month stats:
      {month: {"file": path or None, "ticks": N, "bars_1m": M}}
    """
    diagnostics[symbol] = {}
    parts = []

    for m in months:
        rec = {"file": None, "ticks": 0, "bars_1m": 0}
        files = _candidate_files(inputs_dir, symbol, m)
        if not files:
            diagnostics[symbol][m] = rec
            continue

        path = files[0]       # take the first match
        rec["file"] = path

        try:
            ticks = read_ticks(path)
            rec["ticks"] = int(len(ticks))
            bars = ticks_to_ohlcv_1m(ticks, atr_window=atr_window)
            rec["bars_1m"] = int(len(bars))
            parts.append(bars)
        except Exception as e:
            # leave bars_1m=0; file recorded; continue to next month
            rec["error"] = str(e)

        diagnostics[symbol][m] = rec

    if not parts:
        return pd.DataFrame(columns=["open","high","low","close","volume","atr"])

    df1m = pd.concat(parts, axis=0).sort_index()
    df1m = df1m[~df1m.index.duplicated(keep="last")]
    return df1m

