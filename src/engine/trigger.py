import pandas as pd
import numpy as np

from .adaptive import AdaptiveController
from .utils import zscore_logret, body_dom


class Trigger:
    def __init__(self, cfg: dict, df1m: pd.DataFrame, atr1m: pd.Series, ac: AdaptiveController):
        self.cfg = cfg
        self.df1m = df1m
        self.atr1m = atr1m.replace(0, 1e-9)
        self.ac = ac

        win = int(cfg['entry']['momentum']['zscore_window'])

        # Precompute z-score of 1m log-returns once
        self._zret = zscore_logret(df1m['close'], win)

        # Precompute True Range / ATR once
        prev_c = df1m['close'].shift(1)
        tr_series = pd.concat([
            (df1m['high'] - df1m['low']),
            (df1m['high'] - prev_c).abs(),
            (df1m['low']  - prev_c).abs()
        ], axis=1).max(axis=1)
        self._tr_over_atr = tr_series / self.atr1m

    def power_bar_ok(self, ts: pd.Timestamp, i_bar_1m: int) -> dict:
        tp = self.ac.trigger_params(i_bar_1m)
        z_k = tp['zscore_k']
        rng_min = tp['range_atr_min']

        win = int(self.cfg['entry']['momentum']['zscore_window'])
        min_body = float(self.cfg['entry']['momentum']['min_body_dom'])

        if i_bar_1m < win + 2:
            return {'ok': False, 'z_k': z_k, 'range_atr_min': rng_min}

        last = self.df1m.iloc[i_bar_1m]
        body = float(body_dom(last))
        zret = float(self._zret.iat[i_bar_1m])
        tratr = float(self._tr_over_atr.iat[i_bar_1m])

        ok = (abs(zret) >= z_k) and (body >= min_body) and (tratr >= rng_min)
        return {
            'ok': bool(ok),
            'z_k': z_k,
            'range_atr_min': rng_min,
            'zret': zret,
            'body_dom': body,
            'tr_atr': tratr,
        }

