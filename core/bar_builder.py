# core/bar_builder.py
import pandas as pd
import numpy as np

def _infer_ts_series(df):
    for c in ["timestamp","ts","time","T"]:
        if c in df.columns:
            return df[c]
    raise ValueError("Tick CSV must have a 'timestamp' column")

def read_ticks_to_bars(path: str, bar_minutes: int = 1) -> pd.DataFrame:
    """
    Expect csv with columns: timestamp (ms or ISO), price, qty
    """
    df = pd.read_csv(path)
    ts = _infer_ts_series(df)
    if np.issubdtype(ts.dtype, np.number):
        dt = pd.to_datetime(ts, unit='ms', utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True)
    df["_dt"] = dt.dt.floor(f"{bar_minutes}min")
    # price column
    px_col = None
    for c in ["price","px","p"]:
        if c in df.columns:
            px_col = c
            break
    if px_col is None:
        raise ValueError("Tick CSV must have a 'price' column")
    qty_col = "qty" if "qty" in df.columns else None

    grouped = df.groupby("_dt")
    o = grouped[px_col].first()
    h = grouped[px_col].max()
    l = grouped[px_col].min()
    c = grouped[px_col].last()
    v = grouped[qty_col].sum() if qty_col else (grouped.size().astype(float))

    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).reset_index()
    bars = bars.rename(columns={"_dt":"ts"}).sort_values("ts").reset_index(drop=True)

    # ATR
    tr = np.maximum(bars["high"]-bars["low"],
                    np.maximum((bars["high"]-bars["close"].shift()).abs(),
                               (bars["low"]-bars["close"].shift()).abs()))
    bars["atr"] = tr.rolling(14, min_periods=1).mean()
    return bars
