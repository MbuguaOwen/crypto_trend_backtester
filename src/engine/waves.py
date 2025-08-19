
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .utils import atr, resample_ohlcv

class WaveGate:
    """
    Lean, causal W1/W2 detector using ATR-ZigZag on 5m.
    Produces:
      - armed: bool
      - W2_high / W2_low: levels to breach for momentum ignition
    """
    def __init__(self, cfg: dict):
        wz = cfg['waves']['zigzag']
        th = cfg['waves']['thresholds']
        self.atr_window = int(wz['atr_window'])
        self.atr_mults = list(wz['atr_mults'])
        self.max_lookback_bars = int(wz['max_lookback_bars'])
        self.min_conf = float(th['min_confidence'])
        self.w2_post_min = float(th['w2_posterior_min'])
        self.w2_end_arm = float(th['w2_end_arm'])
        self.max_age_impulse_bars = int(th['max_age_impulse_bars'])

    @staticmethod
    def _zigzag_atr(df5: pd.DataFrame, atr_series: pd.Series, mult: float):
        pivots = []  # list of (idx, price, type) type ∈ {'H','L'}
        if df5.empty:
            return pivots
        last_type = None
        last_price = df5['close'].iloc[0]
        last_idx = df5.index[0]

        for i in range(1, len(df5)):
            row = df5.iloc[i]
            idx = df5.index[i]
            a = float(atr_series.iat[i])
            if not np.isfinite(a) or a <= 0:
                continue
            if last_type in (None, 'L'):  # looking for H
                if row['high'] >= last_price + mult * a:
                    # flip to H at this bar's high
                    pivots.append((idx, row['high'], 'H'))
                    last_type = 'H'
                    last_price = row['high']
                    last_idx = idx
                else:
                    # update potential low
                    if row['low'] < last_price:
                        last_price = row['low']
                        last_idx = idx
                        if last_type is None:
                            pivots = [(idx, row['low'], 'L')]
                            last_type = 'L'
            elif last_type == 'H':  # looking for L
                if row['low'] <= last_price - mult * a:
                    pivots.append((idx, row['low'], 'L'))
                    last_type = 'L'
                    last_price = row['low']
                    last_idx = idx
                else:
                    if row['high'] > last_price:
                        last_price = row['high']
                        last_idx = idx
        # ensure first pivot exists
        if not pivots:
            pivots = [(df5.index[0], df5['low'].iloc[0], 'L')]
        return pivots

    def _w1_w2_from_pivots(self, pivots):
        """
        Find most recent (L,H,L) for long or (H,L,H) for short.
        Return dict with structure.
        """
        if len(pivots) < 3:
            return None
        # work from the end
        for i in range(len(pivots)-1, 1, -1):
            p2 = pivots[i]     # last
            p1 = pivots[i-1]
            p0 = pivots[i-2]
            patt = (p0[2], p1[2], p2[2])
            if patt == ('L','H','L'):  # long candidate
                return {'dir':'LONG',
                        'w1_start': p0, 'w1_end': p1, 'w2_end': p2}
            if patt == ('H','L','H'):  # short candidate
                return {'dir':'SHORT',
                        'w1_start': p0, 'w1_end': p1, 'w2_end': p2}
        return None

    def _score_w2_end(self, df5: pd.DataFrame, w):
        """Score W2 termination: depth fit, compression, time symmetry. 0..1"""
        (t0, p0, _), (t1, p1, _), (t2, p2, _) = w['w1_start'], w['w1_end'], w['w2_end']
        if w['dir'] == 'LONG':
            w1_low, w1_high = p0, p1
            w2_low = p2
            depth = (w1_high - w2_low) / max(1e-9, (w1_high - w1_low))
        else:
            w1_high, w1_low = p0, p1
            w2_high = p2
            depth = (w2_high - w1_low) / max(1e-9, (w1_high - w1_low))

        # depth score
        depth_score = 0.0
        if 0.50 <= depth <= 0.618:
            depth_score = 1.0
        elif 0.618 < depth <= 0.786:
            depth_score = 0.7

        # compression: ATR3 / medianATR20
        atr3 = atr(df5, 3).iloc[-1]
        med20 = atr(df5, 20).iloc[-20:].median()
        comp_ratio = float(atr3 / max(1e-9, med20))
        comp_score = 1.0 if comp_ratio <= 0.7 else (0.0 if comp_ratio >= 1.0 else (1.0 - (comp_ratio-0.7)/0.3))

        # time symmetry
        bars_w1 = max(1, df5.index.get_loc(t1) - df5.index.get_loc(t0))
        bars_w2 = max(1, df5.index.get_loc(t2) - df5.index.get_loc(t1))
        time_score = 1.0 if bars_w2 <= 1.5 * bars_w1 else (0.0 if bars_w2 >= 3.0 * bars_w1 else 0.5)

        # weighted
        score = 0.30*depth_score + 0.20*comp_score + 0.10*time_score + 0.40*0.75  # base momentum prior
        # posterior/confidence (lean)
        posterior = 0.6*depth_score + 0.4*comp_score
        conf = (depth_score + comp_score + time_score)/3.0
        return float(score), float(posterior), float(conf), depth

    def compute_at(self, df5_all: pd.DataFrame, atr5_all: pd.Series, ts):
        # pad to last 5m bar at/<= ts
        if ts < df5_all.index[0]:
            return {'armed': False}
        # Find the most recent 5m bar ≤ ts
        j = df5_all.index.get_indexer([ts], method='pad')[0]
        if j == -1:
            return {'armed': False}
        start = max(0, j - self.max_lookback_bars)
        df5 = df5_all.iloc[start:j+1]
        atr5 = atr5_all.iloc[start:j+1]
        if len(df5) < 60:
            return {'armed': False}

        pivots = self._zigzag_atr(df5, atr5, self.atr_mults[0])
        w = self._w1_w2_from_pivots(pivots)
        if not w:
            return {'armed': False}

        score, posterior, conf, depth = self._score_w2_end(df5, w)

        armed = (score >= self.w2_end_arm) and (posterior >= self.w2_post_min) and (conf >= self.min_conf)

        # Levels
        if w['dir'] == 'LONG':
            W2_high = w['w1_end'][1]
            W2_low  = w['w2_end'][1]
        else:
            W2_low  = w['w1_end'][1]
            W2_high = w['w2_end'][1]

        # Age check
        pos_end = df5.index.get_indexer([w['w1_end'][0]])[0]
        if pos_end == -1:
            armed = False
            pos_end = len(df5) - 1
        age_bars = (len(df5) - 1) - pos_end
        if age_bars > self.max_age_impulse_bars:
            armed = False

        return {
            'armed': bool(armed),
            'dir': w['dir'],
            'W2_high': float(W2_high),
            'W2_low': float(W2_low),
            'score': float(score),
            'posterior': float(posterior),
            'confidence': float(conf),
            'depth': float(depth),
        }
