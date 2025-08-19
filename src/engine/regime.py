
import pandas as pd
from .utils import resample_ohlcv

LONG, SHORT, FLAT = "LONG","SHORT","FLAT"

class TSMOMRegime:
    def __init__(self, cfg: dict):
        self.tfs = cfg['regime']['timeframes']
        self.require = int(cfg['regime']['vote']['require'])

    def decide(self, df1m: pd.DataFrame) -> str:
        votes_long = 0
        votes_short = 0
        for tf, params in self.tfs.items():
            lb = int(params['lookback_closes'])
            if tf == "1m":
                closes = df1m['close']
            elif tf == "5m":
                closes = resample_ohlcv(df1m, '5min')['close']
            elif tf == "15m":
                closes = resample_ohlcv(df1m, '15min')['close']
            elif tf == "1h":
                closes = resample_ohlcv(df1m, '1H')['close']
            else:
                continue
            if len(closes) <= lb:
                continue
            mom = closes.iloc[-1] / closes.shift(lb).iloc[-1] - 1.0
            if mom > 0:
                votes_long += 1
            elif mom < 0:
                votes_short += 1
        if votes_long >= self.require and votes_long > votes_short:
            return LONG
        if votes_short >= self.require and votes_short > votes_long:
            return SHORT
        return FLAT
