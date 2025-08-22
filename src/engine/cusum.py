import numpy as np
import pandas as pd


def build_cusum_bars(df1m: pd.DataFrame, kappa_series: pd.Series) -> pd.DataFrame:
    """
    Build event-time OHLCV using CUSUM on close-to-close Δ.
    - df1m: index tz-aware, 1m OHLCV with columns ['open','high','low','close','volume'].
    - kappa_series: pd.Series aligned to df1m.index with per-bar κ (>=0).
    Returns: df_event with same columns, index at the *end time* of each event bar.
    Causal: only completes a bar when |cumΔ| >= κ, then resets from that bar.
    """
    assert df1m.index.equals(kappa_series.index)
    o, h, l, v = None, None, None, 0.0
    cum = 0.0
    last_close = None
    rows = []
    for ts, row in df1m.iterrows():
        c = float(row['close'])
        if last_close is None:
            last_close = c
            o = float(row['open'])
            h = float(row['high'])
            l = float(row['low'])
            v = float(row['volume'])
            continue
        delta = c - last_close
        last_close = c

        h = max(h, float(row['high']))
        l = min(l, float(row['low']))
        v += float(row['volume'])

        cum += delta
        k = float(kappa_series.loc[ts])
        if k <= 0:
            continue

        if abs(cum) >= k:
            rows.append((ts, o, h, l, c, v))
            o, h, l, v = c, c, c, 0.0
            cum = 0.0

    if not rows:
        return pd.DataFrame(columns=['open','high','low','close','volume'], index=pd.DatetimeIndex([], tz=df1m.index.tz))
    out = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume']).set_index('ts')
    out.index = pd.DatetimeIndex(out.index, tz=df1m.index.tz)
    return out
