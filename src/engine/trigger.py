import pandas as pd

from .adaptive import AdaptiveController
from .utils import zscore_logret, body_dom, true_range_last


class Trigger:
    def __init__(self, cfg: dict, df1m: pd.DataFrame, atr1m: pd.Series, ac: AdaptiveController):
        self.cfg = cfg
        self.df1m = df1m
        self.atr1m = atr1m
        self.ac = ac

    def power_bar_ok(self, ts: pd.Timestamp, i_bar_1m: int) -> dict:
        tp = self.ac.trigger_params(i_bar_1m)
        z_k = tp['zscore_k']
        rng_min = tp['range_atr_min']

        win = int(self.cfg['entry']['momentum']['zscore_window'])
        min_body = float(self.cfg['entry']['momentum']['min_body_dom'])

        if i_bar_1m < win + 2:
            return {'ok': False, 'z_k': z_k, 'range_atr_min': rng_min}

        df_cut = self.df1m.iloc[:i_bar_1m + 1]
        zret = zscore_logret(df_cut['close'], win).iloc[-1]
        last = df_cut.iloc[-1]
        prev_close = df_cut['close'].iloc[-2]
        body = body_dom(last)
        tr = true_range_last(last, prev_close)
        atr = self.atr1m.iloc[i_bar_1m]
        tratr = tr / max(1e-9, atr)

        ok = (abs(zret) >= z_k) and (body >= min_body) and (tratr >= rng_min)
        return {
            'ok': bool(ok),
            'z_k': z_k,
            'range_atr_min': rng_min,
            'zret': float(zret),
            'body_dom': float(body),
            'tr_atr': float(tratr),
        }

