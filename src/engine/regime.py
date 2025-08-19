
import pandas as pd
from .utils import resample_ohlcv

LONG, SHORT, FLAT = "LONG","SHORT","FLAT"

class TSMOMRegime:
    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.tfs = cfg['regime']['timeframes']
        self.require = int(cfg['regime']['vote']['require'])
        # Precompute closes once (no look-ahead in decide_at)
        self.closes = {
            "1m": df1m['close'].copy()
        }
        self.closes["5m"]  = resample_ohlcv(df1m, '5min')['close']
        self.closes["15m"] = resample_ohlcv(df1m, '15min')['close']
        self.closes["1h"]  = resample_ohlcv(df1m, '1h')['close']

    def decide_at(self, ts) -> str:
        votes_long = 0
        votes_short = 0
        for tf, params in self.tfs.items():
            lb = int(params['lookback_closes'])
            if tf not in self.closes:
                continue
            closes = self.closes[tf]
            # As-of TS (pad to last known bar ≤ ts)
            if ts < closes.index[0]:
                continue
            c_now = closes[:ts].iloc[-1]
            if len(closes[:ts]) <= lb:
                continue
            c_then = closes[:ts].shift(lb).dropna().iloc[-1]
            mom = c_now / c_then - 1.0
            if mom > 0:
                votes_long += 1
            elif mom < 0:
                votes_short += 1
        if votes_long >= self.require and votes_long > votes_short:
            return LONG
        if votes_short >= self.require and votes_short > votes_long:
            return SHORT
        return FLAT
