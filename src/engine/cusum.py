import numpy as np
import pandas as pd


def build_cusum_bars(df1m: pd.DataFrame, kappa: pd.Series) -> pd.DataFrame:
    """
    Build event-time OHLCV bars from 1m OHLCV via CUSUM on close-to-close deltas.
    κ (kappa) must be a Series aligned to df1m.index (tz-aware UTC).
    Emits bars with index = event end timestamps (tz-aware UTC).
    Causal: only completes a bar when |cumΔ| >= κ at that bar, then resets.
    """
    assert df1m.index.equals(kappa.index), "kappa must align to df1m index"

    rows = []
    open_, high_, low_, vol_ = None, None, None, 0.0
    last_close = None
    cum = 0.0

    for ts, row in df1m.iterrows():
        close = float(row['close'])
        if last_close is None:
            last_close = close
            open_ = float(row['open']); high_ = float(row['high'])
            low_ = float(row['low']);  vol_  = float(row['volume'])
            continue

        delta = close - last_close
        last_close = close

        # roll OHLCV for the current (open) segment
        high_ = max(high_, float(row['high']))
        low_  = min(low_,  float(row['low']))
        vol_ += float(row['volume'])

        k = float(kappa.loc[ts])
        if k <= 0:
            continue

        cum += delta
        if abs(cum) >= k:
            # complete an event bar at this timestamp
            rows.append((ts, open_, high_, low_, close, vol_))
            # reset segment (starting next bar)
            open_, high_, low_, vol_ = close, close, close, 0.0
            cum = 0.0

    if not rows:
        return pd.DataFrame(columns=['open','high','low','close','volume'],
                            index=pd.DatetimeIndex([], tz=df1m.index.tz))
    out = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume']).set_index('ts')
    out.index = pd.DatetimeIndex(out.index, tz=df1m.index.tz)  # ensure tz-aware
    return out
