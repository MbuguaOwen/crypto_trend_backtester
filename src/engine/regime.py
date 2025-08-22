import numpy as np
import pandas as pd

TF_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h":   "1h",
    "5h":   "5h",
}


def _resample(df1m, tf):
    if tf == "1min":
        return df1m
    rule = TF_MAP[tf].lower()
    return df1m.resample(rule, label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()


class TSMOMRegime:
    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.cfg = cfg
        self.tf_specs = cfg['regime']['ts_mom']['timeframes']
        self.require = int(cfg['regime']['ts_mom']['require_majority'])

        # pre-resample once
        self.frames = {spec['tf']: _resample(df1m, spec['tf']) for spec in self.tf_specs}

        # precompute per-TF majority sign and strength arrays
        self._prep = {}
        for spec in self.tf_specs:
            tf = spec['tf']
            k = int(spec.get('lookback_closes', 3))
            df = self.frames[tf]
            close = df['close']

            # matrix of last k percentage changes vs last close (vectorized)
            cols = [(close / close.shift(i) - 1.0) for i in range(1, k + 1)]
            pct = pd.concat(cols, axis=1)

            signs = np.sign(pct.values)                # shape (n, k)
            maj = np.sign(signs.sum(axis=1)).astype(int)
            maj_series = pd.Series(maj, index=df.index)

            # per-row z of those k pct changes; strength = mean |z|
            row_mu = pct.mean(axis=1)
            row_sd = pct.std(axis=1).replace(0, np.nan)
            z = (pct.sub(row_mu, axis=0)).div(row_sd, axis=0)
            strength = z.abs().mean(axis=1).fillna(0.0)

            self._prep[tf] = {'maj': maj_series, 'str': strength}

    def compute_at(self, ts) -> dict:
        votes = []
        strength_parts = []

        for spec in self.tf_specs:
            tf = spec['tf']
            df = self.frames[tf]
            if ts < df.index[0]:
                return {'dir': 'FLAT', 'score': 0.0, 'strength': 0.0}
            idx = df.index.get_indexer([ts], method='pad')[0]
            if idx == -1:
                return {'dir': 'FLAT', 'score': 0.0, 'strength': 0.0}

            votes.append(int(self._prep[tf]['maj'].iat[idx]))
            strength_parts.append(float(self._prep[tf]['str'].iat[idx]))

        bulls = sum(1 for v in votes if v > 0)
        bears = sum(1 for v in votes if v < 0)
        score = (bulls - bears) / max(1, len(self.tf_specs))

        dir_ = 'FLAT'
        if bulls >= self.require:
            dir_ = 'BULL'
        elif bears >= self.require:
            dir_ = 'BEAR'

        strength = float(np.median(strength_parts)) if strength_parts else 0.0
        return {'dir': dir_, 'score': float(score), 'strength': strength}
