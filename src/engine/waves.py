
import numpy as np
import pandas as pd
from .utils import atr_vec


class WaveGate:
    """Incremental ATR-ZigZag gate on 5m data."""

    def __init__(self, cfg: dict, df5: pd.DataFrame, atr5: pd.Series):
        wz = cfg['waves']['zigzag']
        th = cfg['waves']['thresholds']
        self.atr_mults = list(wz['atr_mults'])
        self.min_conf = float(th['min_confidence'])
        self.w2_post_min = float(th['w2_posterior_min'])
        self.w2_end_arm = float(th['w2_end_arm'])
        self.max_age_impulse_bars = int(th['max_age_impulse_bars'])
        self.df5 = df5
        self.atr5 = atr5
        self.idx_map = {ts: i for i, ts in enumerate(df5.index)}
        self._precompute()

    def _w1_w2_from_pivots(self, pivots):
        if len(pivots) < 3:
            return None
        for i in range(len(pivots) - 1, 1, -1):
            p2 = pivots[i]
            p1 = pivots[i - 1]
            p0 = pivots[i - 2]
            patt = (p0[2], p1[2], p2[2])
            if patt == ('L', 'H', 'L'):
                return {'dir': 'LONG', 'w1_start': p0, 'w1_end': p1, 'w2_end': p2}
            if patt == ('H', 'L', 'H'):
                return {'dir': 'SHORT', 'w1_start': p0, 'w1_end': p1, 'w2_end': p2}
        return None

    def _score_w2_end(self, i: int, w, atr3, atr20):
        (t0, p0, _), (t1, p1, _), (t2, p2, _) = w['w1_start'], w['w1_end'], w['w2_end']
        if w['dir'] == 'LONG':
            w1_low, w1_high = p0, p1
            w2_low = p2
            depth = (w1_high - w2_low) / max(1e-9, (w1_high - w1_low))
        else:
            w1_high, w1_low = p0, p1
            w2_high = p2
            depth = (w2_high - w1_low) / max(1e-9, (w1_high - w1_low))

        depth_score = 0.0
        if 0.50 <= depth <= 0.618:
            depth_score = 1.0
        elif 0.618 < depth <= 0.786:
            depth_score = 0.7

        comp_ratio = atr3[i] / max(1e-9, np.median(atr20[max(0, i-19):i+1]))
        comp_score = 1.0 if comp_ratio <= 0.7 else (0.0 if comp_ratio >= 1.0 else (1.0 - (comp_ratio-0.7)/0.3))

        bars_w1 = max(1, self.idx_map[t1] - self.idx_map[t0])
        bars_w2 = max(1, self.idx_map[t2] - self.idx_map[t1])
        time_score = 1.0 if bars_w2 <= 1.5 * bars_w1 else (0.0 if bars_w2 >= 3.0 * bars_w1 else 0.5)

        score = 0.30 * depth_score + 0.20 * comp_score + 0.10 * time_score + 0.40 * 0.75
        posterior = 0.6 * depth_score + 0.4 * comp_score
        conf = (depth_score + comp_score + time_score) / 3.0
        return float(score), float(posterior), float(conf), depth

    def _precompute(self):
        df5 = self.df5
        atr5 = self.atr5
        high = df5['high'].to_numpy()
        low = df5['low'].to_numpy()
        close = df5['close'].to_numpy()
        atr3 = atr_vec(high, low, close, 3)
        atr20 = atr_vec(high, low, close, 20)
        pivots = []
        last_type = None
        last_price = close[0]
        states = []
        for i in range(len(df5)):
            h = high[i]
            l = low[i]
            c = close[i]
            idx = df5.index[i]
            a = float(atr5.iat[i])
            if not np.isfinite(a) or a <= 0:
                states.append({'armed': False})
                continue
            if last_type in (None, 'L'):
                if h >= last_price + self.atr_mults[0] * a:
                    pivots.append((idx, h, 'H'))
                    last_type = 'H'
                    last_price = h
                else:
                    if l < last_price:
                        last_price = l
                        if last_type is None:
                            pivots = [(idx, l, 'L')]
                            last_type = 'L'
            elif last_type == 'H':
                if l <= last_price - self.atr_mults[0] * a:
                    pivots.append((idx, l, 'L'))
                    last_type = 'L'
                    last_price = l
                else:
                    if h > last_price:
                        last_price = h
            if not pivots:
                pivots = [(idx, l, 'L')]
            w = self._w1_w2_from_pivots(pivots)
            if not w:
                states.append({'armed': False})
                continue
            score, posterior, conf, depth = self._score_w2_end(i, w, atr3, atr20)
            armed = (score >= self.w2_end_arm) and (posterior >= self.w2_post_min) and (conf >= self.min_conf)
            if w['dir'] == 'LONG':
                W2_high = w['w1_end'][1]
                W2_low = w['w2_end'][1]
            else:
                W2_low = w['w1_end'][1]
                W2_high = w['w2_end'][1]
            pos_end = self.idx_map.get(w['w1_end'][0], i)
            age_bars = i - pos_end
            if age_bars > self.max_age_impulse_bars:
                armed = False
            states.append({
                'armed': bool(armed),
                'dir': w['dir'],
                'W2_high': float(W2_high),
                'W2_low': float(W2_low),
                'score': float(score),
                'posterior': float(posterior),
                'confidence': float(conf),
                'depth': float(depth),
            })
        self.states = states

    def compute_at(self, ts):
        j = self.df5.index.get_indexer([ts], method='pad')[0]
        if j == -1:
            return {'armed': False}
        return self.states[j]
