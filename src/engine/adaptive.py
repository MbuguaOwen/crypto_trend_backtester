from dataclasses import dataclass
import numpy as np
import pandas as pd


def pct_rank(series: pd.Series, win: int) -> float:
    s = series.tail(win)
    x = s.values
    last, prev = x[-1], x[:-1]
    return float((prev <= last).mean())


def lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


@dataclass
class WaveParams:
    atr_mult0: float
    atr_window: int
    w2_end_arm: float
    w2_post_min: float
    min_conf: float
    max_age_bars: int


class AdaptiveController:
    """
    Single source of truth for adaptive parameters.
    No fallbacks: Backtest will warm-start until ready; then every value comes from ranges.
    """

    def __init__(self, cfg: dict, atr1m: pd.Series):
        self.cfg = cfg
        self.atr1m = atr1m
        self.state = {
            'w2_scores': [],
            'w2_posts': [],
            'w2_confs': [],
            'w2_durs': [],
            'last_tune_i': -10 ** 9,
        }

    # Called by backtest to ensure readiness BEFORE first trade bar
    def ready(self, i_bar: int, w2_candidates: int) -> bool:
        wm = int(self.cfg['engine']['warmup']['min_1m_bars'])
        wc = int(self.cfg['engine']['warmup']['min_w2_candidates'])
        return (i_bar >= wm) and (w2_candidates >= wc)

    def record_w2(self, score, post, conf, dur_bars):
        self.state['w2_scores'].append(float(score))
        self.state['w2_posts'].append(float(post))
        self.state['w2_confs'].append(float(conf))
        self.state['w2_durs'].append(int(dur_bars))

    # ---- Waves ----
    def waves_params(self, i_bar: int) -> WaveParams:
        ad = self.cfg['waves']['adaptive']
        vol_win = int(ad['vol_window_1m'])
        atr_norm = (self.atr1m / self.atr1m.rolling(vol_win).median()).fillna(method='ffill')
        vol_pctl = pct_rank(atr_norm.iloc[:i_bar + 1], vol_win)

        atr_mult0 = lerp(ad['atr_mult_range'][0], ad['atr_mult_range'][1], vol_pctl)
        atr_window = int(lerp(ad['atr_window_range'][0], ad['atr_window_range'][1], vol_pctl))

        q = float(ad['thresholds_quantile'])
        w2_end_arm = float(np.nanquantile(self.state['w2_scores'][-1000:], q))
        w2_post_min = float(np.nanquantile(self.state['w2_posts'][-1000:], q))
        min_conf = float(np.nanquantile(self.state['w2_confs'][-1000:], q))

        d = np.array(self.state['w2_durs'][-500:], dtype=float)
        med = float(np.nanmedian(d))
        max_age = int(np.clip(med * 1.25, ad['max_age_bars_range'][0], ad['max_age_bars_range'][1]))

        return WaveParams(atr_mult0, atr_window, w2_end_arm, w2_post_min, min_conf, max_age)

    # ---- Trigger ----
    def trigger_params(self, i_bar: int) -> dict:
        ad = self.cfg['entry']['adaptive']
        win = int(self.cfg['waves']['adaptive']['vol_window_1m'])
        atr_norm = (self.atr1m / self.atr1m.rolling(win).median()).fillna(method='ffill')
        vol_pctl = pct_rank(atr_norm.iloc[:i_bar + 1], win)
        return {
            'zscore_k': float(lerp(ad['zscore_k_range'][0], ad['zscore_k_range'][1], vol_pctl)),
            'range_atr_min': float(lerp(ad['range_atr_min_range'][0], ad['range_atr_min_range'][1], vol_pctl)),
        }

    # ---- Risk ----
    def risk_params(self, i_bar: int) -> dict:
        ad = self.cfg['risk']['adaptive']
        win = int(self.cfg['waves']['adaptive']['vol_window_1m'])
        atr_norm = (self.atr1m / self.atr1m.rolling(win).median()).fillna(method='ffill')
        vov = atr_norm.rolling(win // 2).std().fillna(0.0)
        t = pct_rank(vov.iloc[:i_bar + 1], win)
        return {
            'be_trigger_r': float(lerp(ad['be_trigger_r_range'][0], ad['be_trigger_r_range'][1], t)),
            'tsl_start_r': float(lerp(ad['tsl_start_r_range'][0], ad['tsl_start_r_range'][1], t)),
            'tsl_atr_mult': float(lerp(ad['tsl_atr_mult_range'][0], ad['tsl_atr_mult_range'][1], t)),
        }

