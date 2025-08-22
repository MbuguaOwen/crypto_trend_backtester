import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True, fastmath=True)
def _cusum_core(close, high, low, volume, kappa):
    rows_ts = []
    rows_o = []
    rows_h = []
    rows_l = []
    rows_c = []
    rows_v = []

    n = close.shape[0]
    if n == 0:
        return (np.empty((0,), dtype=np.int64),
                np.empty((0,)), np.empty((0,)), np.empty((0,)),
                np.empty((0,)), np.empty((0,)))

    last_close = close[0]
    open_ = close[0]
    high_ = high[0]
    low_  = low[0]
    vol_  = volume[0]
    cum = 0.0

    for i in range(1, n):
        c = close[i]
        delta = c - last_close
        last_close = c

        if high[i] > high_:
            high_ = high[i]
        if low[i] < low_:
            low_ = low[i]
        vol_ += volume[i]

        k = kappa[i]
        if k <= 0.0:
            continue

        cum += delta
        if abs(cum) >= k:
            rows_ts.append(i)
            rows_o.append(open_)
            rows_h.append(high_)
            rows_l.append(low_)
            rows_c.append(c)
            rows_v.append(vol_)

            open_ = c
            high_ = c
            low_  = c
            vol_  = 0.0
            cum   = 0.0

    return (np.array(rows_ts, dtype=np.int64),
            np.array(rows_o), np.array(rows_h), np.array(rows_l),
            np.array(rows_c), np.array(rows_v))


def build_cusum_bars(df1m: pd.DataFrame, kappa: pd.Series) -> pd.DataFrame:
    assert df1m.index.equals(kappa.index), "kappa must align to df1m index"

    close = df1m['close'].to_numpy(dtype=np.float64)
    high  = df1m['high'].to_numpy(dtype=np.float64)
    low   = df1m['low'].to_numpy(dtype=np.float64)
    vol   = df1m['volume'].to_numpy(dtype=np.float64)
    kap   = kappa.to_numpy(dtype=np.float64)

    idxs, o, h, l, c, v = _cusum_core(close, high, low, vol, kap)
    if idxs.size == 0:
        return pd.DataFrame(columns=['open','high','low','close','volume'],
                            index=pd.DatetimeIndex([], tz=df1m.index.tz))

    ts = df1m.index.values[idxs]
    out = pd.DataFrame(
        {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v},
        index=pd.DatetimeIndex(ts, tz=df1m.index.tz)
    )
    return out

