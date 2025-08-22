
import numpy as np
import pandas as pd
from .utils import resample_ohlcv

LONG, SHORT, FLAT = "LONG", "SHORT", "FLAT"


class TSMOMRegime:
    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.tfs = cfg['regime']['timeframes']
        self.require = int(cfg['regime']['vote']['require'])
        self.index = df1m.index
        self.regime_vec = self._precompute(df1m)

    def _precompute(self, df1m: pd.DataFrame):
        n = len(df1m)
        votes_long = np.zeros(n, dtype=np.int8)
        votes_short = np.zeros(n, dtype=np.int8)
        for tf, params in self.tfs.items():
            lb = int(params['lookback_closes'])
            if tf == "1m":
                closes = df1m['close']
            else:
                closes = resample_ohlcv(df1m, tf)['close']
            mom = closes / closes.shift(lb) - 1.0
            sign = np.sign(mom).reindex(df1m.index, method='pad').fillna(0).to_numpy(dtype=np.int8)
            votes_long += (sign > 0).astype(np.int8)
            votes_short += (sign < 0).astype(np.int8)
        regime = np.full(n, FLAT, dtype=object)
        cond_long = (votes_long >= self.require) & (votes_long > votes_short)
        cond_short = (votes_short >= self.require) & (votes_short > votes_long)
        regime[cond_long] = LONG
        regime[cond_short] = SHORT
        return regime

    def decide_at(self, ts) -> str:
        j = self.index.get_indexer([ts], method='pad')[0]
        if j == -1:
            return FLAT
        return self.regime_vec[j]
