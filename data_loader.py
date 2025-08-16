import os, glob
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

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
    # NOTE: position=1 so it renders under the outer "Symbols" bar
    for f in tqdm(files, desc=f"[{symbol}] Loading CSVs", unit="file", dynamic_ncols=True, leave=False, position=1):
        # 1) Read header to map aliases once
        try:
            hdr = pd.read_csv(f, nrows=0, engine="pyarrow")
        except Exception:
            hdr = pd.read_csv(f, nrows=0)
        cols = list(hdr.columns)

        tcol = _find_col(cols, TIME_ALIASES)
        pcol = _find_col(cols, PRICE_ALIASES)
        qcol = _find_col(cols, QTY_ALIASES)
        qqcol = _find_col(cols, QUOTE_QTY_ALIASES)
        mcol = _find_col(cols, MAKER_ALIASES)

        if tcol is None or pcol is None:
            raise ValueError(f"{f}: Missing required columns (found: {cols}). Need timestamp & price aliases.")

        usecols = [tcol, pcol]
        if qcol is not None:
            usecols.append(qcol)
        elif qqcol is not None:
            usecols.append(qqcol)
        if mcol is not None:
            usecols.append(mcol)

        # 2) Stream the file in chunks to show progress
        #    Prefer the default C engine for chunking (pyarrow sometimes returns a full DataFrame).
        try:
            it = pd.read_csv(f, usecols=usecols, chunksize=2_000_000)  # no engine→C engine, chunked
        except Exception:
            # very old pandas fallback
            it = pd.read_csv(f, usecols=usecols, chunksize=2_000_000, low_memory=False)

        buf = []
        # per-file row counter at position=2
        with tqdm(desc=f"[{symbol}] {os.path.basename(f)}", unit="rows", dynamic_ncols=True, leave=False, position=2) as pbar:
            for chunk in it:
                pbar.update(len(chunk))

                df = chunk.rename(columns={tcol: "ts", pcol: "price"})
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

                # Types & hygiene (per chunk)
                df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
                ts_len = int(df["ts"].dropna().astype(str).str.len().median()) if df["ts"].notna().any() else 13
                if ts_len <= 10:
                    df["ts"] = (df["ts"] * 1000).astype("Int64")
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
                df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

                df = df.dropna(subset=["ts","price","qty"])
                df = df[(df["price"] > 0) & (df["qty"] >= 0)]

                # Month filter (UTC)
                ts_dt = pd.to_datetime(df["ts"], unit="ms", utc=True)
                df = df[ts_dt.dt.strftime("%Y-%m").isin(months)]

                if not df.empty:
                    buf.append(df[["ts","price","qty","is_buyer_maker"]])

        if buf:
            parts.append(pd.concat(buf, ignore_index=True))

    if not parts:
        raise FileNotFoundError(f"No rows matched requested months {months} under {path}")

    all_ticks = pd.concat(parts, ignore_index=True)
    all_ticks = (
        all_ticks.sort_values("ts")
                 .drop_duplicates(subset=["ts","price","qty"], keep="last")
                 .reset_index(drop=True)
    )
    return all_ticks

def build_minute_bars(ticks: pd.DataFrame, interval: str = "1min") -> pd.DataFrame:
    if ticks.empty:
        return ticks

    df = ticks.copy()
    df["ts_dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

    # Resample by day for visible progress
    df["day"] = df["ts_dt"].dt.normalize()
    bars = []
    for day, g in tqdm(df.groupby("day"), desc="[Bars] Resampling by day", unit="day", dynamic_ncols=True, leave=False, position=2):
        g = g.set_index("ts_dt")
        o = g["price"].resample(interval).first()
        h = g["price"].resample(interval).max()
        l = g["price"].resample(interval).min()
        c = g["price"].resample(interval).last()
        v = g["qty"].resample(interval).sum()
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
        if not out.empty:
            out["day"] = day
            bars.append(out)

    if not bars:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    out = pd.concat(bars).reset_index()
    out["ts"] = (out["ts_dt"].view("int64") // 1_000_000).astype("int64")
    out = out.drop(columns=["ts_dt","day"]).sort_values("ts").reset_index(drop=True)
    return out[["ts","open","high","low","close","volume"]]
