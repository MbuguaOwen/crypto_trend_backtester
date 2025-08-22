import numpy as np
import pandas as pd

TF_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h":   "1h",   # NOTE: lower-case 'h' to avoid deprecation
    "5h":   "5h",
}

def _resample(df1m, tf):
    if tf == "1min":
        return df1m
    rule = TF_MAP[tf].lower()
    return df1m.resample(rule, label='right', closed='right').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna()

class TSMOMRegime:
    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.cfg = cfg
        self.tf_specs = cfg['regime']['ts_mom']['timeframes']
        self.require = int(cfg['regime']['ts_mom']['require_majority'])
        # pre-resample each TF once
        self.frames = {spec['tf']: _resample(df1m, spec['tf']) for spec in self.tf_specs}
        self._prep = {}
        for spec in self.tf_specs:
            tf = spec['tf']; k = int(spec.get('lookback_closes', 3))
            df = self.frames[tf]
            close = df['close']
            pct = pd.concat([(close / close.shift(i) - 1.0) for i in range(1, k + 1)], axis=1)
            rets_sign = np.sign(pct.values)
            rets_sign = np.nan_to_num(rets_sign, nan=0.0)
            maj = np.sign(rets_sign.sum(axis=1))
            mean = pct.mean(axis=1).values.reshape(-1, 1)
            std = pct.std(axis=1).replace(0, np.nan).values.reshape(-1, 1)
            z = (pct - mean) / std
            strength = np.abs(z).mean(axis=1)
            self._prep[tf] = {
                'maj': pd.Series(maj, index=df.index),
                'str': pd.Series(strength, index=df.index)
            }

    def compute_at(self, ts) -> dict:
        votes = []
        strength_parts = []

        for spec in self.tf_specs:
            tf = spec['tf']
            df = self.frames[tf]
            idx = df.index.get_indexer([ts], method='pad')[0]
            if idx == -1:
                return {'dir':'FLAT','score':0.0,'strength':0.0}
            maj_series = self._prep[tf]['maj']
            str_series = self._prep[tf]['str']
            if idx >= len(maj_series) or pd.isna(maj_series.iat[idx]):
                return {'dir':'FLAT','score':0.0,'strength':0.0}
            votes.append(int(maj_series.iat[idx]))
            strength_parts.append(float(str_series.iat[idx]))

        bulls = sum(1 for v in votes if v > 0)
        bears = sum(1 for v in votes if v < 0)
        score = (bulls - bears) / max(1, len(self.tf_specs))

        dir_ = 'FLAT'
        if bulls >= self.require: dir_ = 'BULL'
        elif bears >= self.require: dir_ = 'BEAR'

        strength = float(np.median(strength_parts)) if strength_parts else 0.0
        return {'dir': dir_, 'score': float(score), 'strength': strength}
