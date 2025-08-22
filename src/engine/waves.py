import math
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .utils import atr


class WaveGate:
    """Wave gate operating on event-time bars."""
    def __init__(self, cfg: dict, df_event: pd.DataFrame, df1m: pd.DataFrame):
        wz = cfg['waves']['zigzag']
        thr = cfg['waves']['thresholds']
        self.cfg = cfg
        self.df_event = df_event
        self.df1m = df1m
        self.atr_window = int(wz['atr_window'])
        self.atr_mults = list(wz['atr_mults'])
        self.max_lookback_ev = int(wz['max_lookback_bars'])
        self.min_cov_minutes = 300   # ≥ 5 hours of 1m coverage

        self.min_conf = float(thr['min_confidence'])
        self.w2_post_min = float(thr['w2_posterior_min'])
        self.w2_end_arm = float(thr['w2_end_arm'])
        self.max_age_impulse_bars = int(thr['max_age_impulse_bars'])

    @staticmethod
    def _zigzag_atr(df: pd.DataFrame, atr_series: pd.Series, mult: float):
        pivots = []
        if df.empty:
            return pivots
        last_type = None
        last_price = df['close'].iloc[0]
        last_idx = df.index[0]
        for i in range(1, len(df)):
            row = df.iloc[i]
            idx = df.index[i]
            a = float(atr_series.iat[i])
            if not np.isfinite(a) or a <= 0:
                continue
            if last_type in (None, 'L'):
                if row['high'] >= last_price + mult * a:
                    pivots.append((idx, row['high'], 'H'))
                    last_type = 'H'
                    last_price = row['high']
                    last_idx = idx
                else:
                    if row['low'] < last_price:
                        last_price = row['low']
                        last_idx = idx
                        if last_type is None:
                            pivots = [(idx, row['low'], 'L')]
                            last_type = 'L'
            elif last_type == 'H':
                if row['low'] <= last_price - mult * a:
                    pivots.append((idx, row['low'], 'L'))
                    last_type = 'L'
                    last_price = row['low']
                    last_idx = idx
                else:
                    if row['high'] > last_price:
                        last_price = row['high']
                        last_idx = idx
        if not pivots:
            pivots = [(df.index[0], df['low'].iloc[0], 'L')]
        return pivots

    def _w1_w2_from_pivots(self, pivots):
        if len(pivots) < 3:
            return None
        for i in range(len(pivots)-1, 1, -1):
            p2 = pivots[i]
            p1 = pivots[i-1]
            p0 = pivots[i-2]
            patt = (p0[2], p1[2], p2[2])
            if patt == ('L','H','L'):
                return {'dir':'LONG','w1_start':p0,'w1_end':p1,'w2_end':p2}
            if patt == ('H','L','H'):
                return {'dir':'SHORT','w1_start':p0,'w1_end':p1,'w2_end':p2}
        return None

    def _score_w2_end(self, df: pd.DataFrame, w: Dict) -> tuple:
        (t0,p0,_),(t1,p1,_),(t2,p2,_) = w['w1_start'], w['w1_end'], w['w2_end']
        if w['dir']=='LONG':
            w1_low, w1_high = p0, p1
            w2_low = p2
            depth = (w1_high - w2_low)/max(1e-9,(w1_high - w1_low))
        else:
            w1_high, w1_low = p0, p1
            w2_high = p2
            depth = (w2_high - w1_low)/max(1e-9,(w1_high - w1_low))

        depth_score = 0.0
        if 0.50 <= depth <= 0.618:
            depth_score = 1.0
        elif 0.618 < depth <= 0.786:
            depth_score = 0.7

        atr3 = atr(df, 3).iloc[-1]
        med20 = atr(df, 20).iloc[-20:].median()
        comp_ratio = float(atr3 / max(1e-9, med20))
        comp_score = 1.0 if comp_ratio <= 0.7 else (0.0 if comp_ratio >= 1.0 else (1.0 - (comp_ratio-0.7)/0.3))

        bars_w1 = max(1, df.index.get_loc(t1) - df.index.get_loc(t0))
        bars_w2 = max(1, df.index.get_loc(t2) - df.index.get_loc(t1))
        time_score = 1.0 if bars_w2 <= 1.5*bars_w1 else (0.0 if bars_w2 >= 3.0*bars_w1 else 0.5)

        score = 0.30*depth_score + 0.20*comp_score + 0.10*time_score + 0.40*0.75
        posterior = 0.6*depth_score + 0.4*comp_score
        conf = (depth_score + comp_score + time_score)/3.0
        return float(score), float(posterior), float(conf), depth

    def _score_w2(self, df: pd.DataFrame, w: Dict):
        score, post, conf, _ = self._score_w2_end(df, w)
        armed = (score >= self.w2_end_arm) and (post >= self.w2_post_min) and (conf >= self.min_conf)
        return armed, float(score), float(post), float(conf)

    def compute_at(self, ts):
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts, utc=True)

        df_e = self.df_event
        if df_e.empty or ts < df_e.index[0]:
            return {'armed': False}

        j = df_e.index.get_indexer([ts], method='pad')[0]
        if j == -1:
            return {'armed': False}

        start = max(0, j - self.max_lookback_ev)
        ev = df_e.iloc[start:j+1]
        if ev.empty or len(ev) < 2:
            return {'armed': False}

        t0 = ev.index[0]
        if not isinstance(t0, pd.Timestamp):
            t0 = pd.to_datetime(t0, utc=True)

        i0 = self.df1m.index.get_indexer([t0], method='backfill')[0]
        i1 = self.df1m.index.get_indexer([ts], method='pad')[0]
        if i0 == -1 or i1 == -1:
            return {'armed': False}
        coverage = (i1 - i0 + 1)
        if coverage < self.min_cov_minutes:
            return {'armed': False}

        tr = (ev['high'] - ev['low']).abs()
        atr_ev = tr.rolling(self.atr_window, min_periods=self.atr_window).mean()
        if atr_ev.isna().all():
            return {'armed': False}

        pivots = self._zigzag_atr(ev, atr_ev, self.atr_mults[0])
        w = self._w1_w2_from_pivots(pivots)
        if w is None:
            return {'armed': False}

        armed, score, post, conf = self._score_w2(ev, w)
        if not armed:
            return {'armed': False}

        pos_end = ev.index.get_indexer([w['w1_end'][0]], method='pad')[0]
        if pos_end == -1:
            return {'armed': False}
        age_bars = (len(ev) - 1) - pos_end
        if age_bars > self.max_age_impulse_bars:
            return {'armed': False}

        if w['dir'] == 'LONG':
            w2_high = w['w1_end'][1]
            w2_low = w['w2_end'][1]
        else:
            w2_low = w['w1_end'][1]
            w2_high = w['w2_end'][1]

        return {
            'armed': True,
            'dir': w['dir'],
            'w2_high': float(w2_high),
            'w2_low': float(w2_low),
            'score': score,
            'posterior': post,
            'confidence': conf,
            'frame': 'event',
        }
