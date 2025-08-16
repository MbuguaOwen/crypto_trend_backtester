
import os, glob
from typing import List, Optional
import numpy as np
import pandas as pd

QTY_ALIASES = ["qty","quantity","amount","size","vol","volume","q","trade_quantity","last_qty"]
QUOTE_QTY_ALIASES = ["quote_qty","quoteQuantity","qv","amount_quote","notional","quoteVolume","quote_vol"]
PRICE_ALIASES = ["price","p","last_price","rate"]
TIME_ALIASES = ["timestamp","ts","time","T","event_time","trade_time_ms","trade_time"]
MAKER_ALIASES = ["is_buyer_maker","isBuyerMaker","maker","is_maker","is_seller_maker","isSellerMaker"]

def _find_col(cols: List[str], aliases: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None

def _coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    mapping = {"true": True, "false": False, "t": True, "f": False, "1": True, "0": False, "buy": False, "sell": True}
    return s.astype(str).str.lower().map(mapping).fillna(False)

def load_ticks_for_months(symbol: str, data_dir: str, months: List[str]) -> pd.DataFrame:
    path = os.path.join(data_dir, symbol)
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")

    parts = []
    for f in files:
        df = pd.read_csv(f)
        cols = list(df.columns)
        tcol = _find_col(cols, TIME_ALIASES)
        pcol = _find_col(cols, PRICE_ALIASES)
        qcol = _find_col(cols, QTY_ALIASES)
        qqcol = _find_col(cols, QUOTE_QTY_ALIASES)
        mcol = _find_col(cols, MAKER_ALIASES)

        if tcol is None or pcol is None:
            raise ValueError(f"{f}: Missing required columns (found: {cols}). Need timestamp & price aliases.")

        df = df.rename(columns={tcol: "ts", pcol: "price"})
        if qcol is not None:
            df = df.rename(columns={qcol: "qty"})
        elif qqcol is not None:
            df["qty"] = pd.to_numeric(df[qqcol], errors="coerce") / pd.to_numeric(df["price"], errors="coerce")
        else:
            df["qty"] = 0.0

        if mcol is not None:
            df["is_buyer_maker"] = _coerce_bool(df[mcol])
        else:
            df["is_buyer_maker"] = False

        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        ts_len = int(np.nanmedian(df["ts"].dropna().astype(str).str.len())) if df["ts"].notna().any() else 13
        if ts_len <= 10:
            df["ts"] = (df["ts"] * 1000).astype("Int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

        df = df.dropna(subset=["ts","price","qty"])
        df = df[(df["price"] > 0) & (df["qty"] >= 0)]
        df = df.sort_values("ts").drop_duplicates(subset=["ts","price","qty"], keep="last")

        ts_dt = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["yyyy_mm"] = ts_dt.dt.strftime("%Y-%m")
        df = df[df["yyyy_mm"].isin(months)].drop(columns=["yyyy_mm"])

        parts.append(df[["ts","price","qty","is_buyer_maker"]])

    if not parts:
        raise FileNotFoundError(f"No rows matched requested months {months} under {path}")

    all_ticks = pd.concat(parts, ignore_index=True)
    all_ticks = all_ticks.sort_values("ts").reset_index(drop=True)
    return all_ticks

def build_minute_bars(ticks: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    if ticks.empty:
        return ticks

    df = ticks.copy()
    df["ts_dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts_dt")

    o = df["price"].resample("1min").first()
    h = df["price"].resample("1min").max()
    l = df["price"].resample("1min").min()
    c = df["price"].resample("1min").last()
    v = df["qty"].resample("1min").sum()

    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    out = out.reset_index()
    out["ts"] = (out["ts_dt"].view("int64") // 1_000_000).astype("int64")
    out = out.drop(columns=["ts_dt"])
    return out[["ts","open","high","low","close","volume"]].sort_values("ts").reset_index(drop=True)
