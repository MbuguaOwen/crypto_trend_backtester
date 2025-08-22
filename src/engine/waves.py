import pandas as pd
import numpy as np

from .adaptive import AdaptiveController, WaveParams


class WaveGate:
    def __init__(self, cfg: dict, df_event: pd.DataFrame, df1m: pd.DataFrame, ac: AdaptiveController):
        self.cfg = cfg
        self.df_event = df_event      # tz-aware event bars
        self.df1m = df1m              # tz-aware 1m bars
        self.ac = ac
        self.max_lookback_ev = int(cfg['waves']['zigzag']['max_lookback_bars'])
        self.min_cov_minutes = 300    # ≥5h underlying coverage

        # Precompute TR and its cumulative sum for O(1) rolling means
        self.ev_tr = (self.df_event['high'] - self.df_event['low']).abs()
        self.ev_tr_cum = self.ev_tr.cumsum()

    # fast rolling mean using cumsum; j_end inclusive
    def _rolling_mean_fast(self, j_end: int, w: int) -> float:
        if w <= 0:
            return float('nan')
        j0 = max(0, j_end - w + 1)
        cum = float(self.ev_tr_cum.iat[j_end]) - (float(self.ev_tr_cum.iat[j0 - 1]) if j0 > 0 else 0.0)
        return cum / float(j_end - j0 + 1)

    # --- PREWARM: walk event bars up to ts_end and record W2 candidates for quantiles
    def prewarm_until(self, ts_end: pd.Timestamp):
        dfe = self.df_event
        if dfe.empty or ts_end < dfe.index[0]:
            return 0
        j = dfe.index.get_indexer([ts_end], method='pad')[0]
        if j == -1:
            return 0
        seen = 0
        step = max(1, len(dfe.iloc[:j + 1]) // 1000)
        for jj in range(0, j + 1, step):
            ev = dfe.iloc[max(0, jj - self.max_lookback_ev): jj + 1]
            if len(ev) < 10:
                continue
            t0 = ev.index[0]
            i0 = self.df1m.index.get_indexer([t0], method='backfill')[0]
            i1 = self.df1m.index.get_indexer([ev.index[-1]], method='pad')[0]
            if i0 == -1 or i1 == -1:
                continue
            if (i1 - i0 + 1) < self.min_cov_minutes:
                continue

            # use median window as prewarm approximation
            aw_min, aw_max = self.cfg['waves']['adaptive']['atr_window_range']
            w = int((aw_min + aw_max) // 2)
            if w <= 0 or len(ev) < w:
                continue

            # compute atr_ev via cumsum fast-path for the small slice
            j_end = self.df_event.index.get_indexer([ev.index[-1]], method='pad')[0]
            j_start = j_end - (len(ev) - 1)
            atr_vals = [self._rolling_mean_fast(j, w) for j in range(j_start, j_end + 1)]
            atr_ev = pd.Series(atr_vals, index=ev.index)
            if atr_ev.isna().all():
                continue

            am_min, am_max = self.cfg['waves']['adaptive']['atr_mult_range']
            piv = self._zigzag_atr(ev, atr_ev, (am_min + am_max) / 2.0)
            wv = self._w1_w2_from_pivots(piv)
            if wv is None:
                continue
            armed, score, post, conf = self._score_w2(ev, wv, 0.62, 0.60, 0.60)
            if not armed:
                continue
            pos_end = ev.index.get_indexer([wv['w1_end'][0]], method='pad')[0]
            if pos_end == -1:
                continue
            age_bars = (len(ev) - 1) - pos_end
            self.ac.record_w2(score, post, conf, age_bars)
            seen += 1
        return seen

    # --- LIVE: compute at 1m ts using strictly adaptive params
    def compute_at(self, ts: pd.Timestamp, i_bar_1m: int):
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts, utc=True)
        dfe = self.df_event
        if dfe.empty or ts < dfe.index[0]:
            return {'armed': False}
        j = dfe.index.get_indexer([ts], method='pad')[0]
        if j == -1:
            return {'armed': False}
        ev = dfe.iloc[max(0, j - self.max_lookback_ev): j + 1]
        if len(ev) < 10:
            return {'armed': False}

        t0 = ev.index[0]
        i0 = self.df1m.index.get_indexer([t0], method='backfill')[0]
        i1 = self.df1m.index.get_indexer([ts], method='pad')[0]
        if i0 == -1 or i1 == -1:
            return {'armed': False}
        if (i1 - i0 + 1) < self.min_cov_minutes:
            return {'armed': False}

        p: WaveParams = self.ac.waves_params(i_bar_1m)

        # fast ATR over event bars via cumsum
        w = int(p.atr_window)
        if w <= 0:
            return {'armed': False}
        j_end = self.df_event.index.get_indexer([ev.index[-1]], method='pad')[0]
        j_start = j_end - (len(ev) - 1)
        atr_vals = [self._rolling_mean_fast(jj, w) for jj in range(j_start, j_end + 1)]
        atr_ev = pd.Series(atr_vals, index=ev.index)
        if atr_ev.isna().all():
            return {'armed': False}

        piv = self._zigzag_atr(ev, atr_ev, p.atr_mult0)
        wv = self._w1_w2_from_pivots(piv)
        if wv is None:
            return {'armed': False}

        armed, score, post, conf = self._score_w2(ev, wv, p.w2_end_arm, p.w2_post_min, p.min_conf)
        if not armed:
            return {'armed': False}

        pos_end = ev.index.get_indexer([wv['w1_end'][0]], method='pad')[0]
        if pos_end == -1:
            return {'armed': False}
        age_bars = (len(ev) - 1) - pos_end
        if age_bars > p.max_age_bars:
            return {'armed': False}

        self.ac.record_w2(score, post, conf, age_bars)

        if wv['dir'] == 'LONG':
            w2_high = wv['w1_end'][1]
            w2_low = wv['w2_end'][1]
        else:
            w2_low = wv['w1_end'][1]
            w2_high = wv['w2_end'][1]

        return {
            'armed': True,
            'dir': wv['dir'],
            'w2_high': w2_high,
            'w2_low': w2_low,
            'score': score,
            'posterior': post,
            'confidence': conf,
            'frame': 'event',
            'params': {
                'atr_mult0': p.atr_mult0,
                'atr_window': p.atr_window,
                'w2_end_arm': p.w2_end_arm,
                'w2_post_min': p.w2_post_min,
                'min_conf': p.min_conf,
                'max_age_bars': p.max_age_bars,
            },
        }

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

    @staticmethod
    def _w1_w2_from_pivots(pivots):
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

    @staticmethod
    def _score_w2(df: pd.DataFrame, w, w2_end_arm: float, w2_post_min: float, min_conf: float):
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

        atr3 = (df['high'] - df['low']).rolling(3).max().iloc[-1]
        med20 = (df['high'] - df['low']).rolling(20).max().iloc[-20:].median()
        comp_ratio = float(atr3 / max(1e-9, med20))
        comp_score = 1.0 if comp_ratio <= 0.7 else (0.0 if comp_ratio >= 1.0 else (1.0 - (comp_ratio - 0.7) / 0.3))

        bars_w1 = max(1, df.index.get_loc(t1) - df.index.get_loc(t0))
        bars_w2 = max(1, df.index.get_loc(t2) - df.index.get_loc(t1))
        time_score = 1.0 if bars_w2 <= 1.5 * bars_w1 else (0.0 if bars_w2 >= 3.0 * bars_w1 else 0.5)

        score = 0.30 * depth_score + 0.20 * comp_score + 0.10 * time_score + 0.40 * 0.75
        posterior = 0.6 * depth_score + 0.4 * comp_score
        conf = (depth_score + comp_score + time_score) / 3.0
        armed = (score >= w2_end_arm) and (posterior >= w2_post_min) and (conf >= min_conf)
        return armed, float(score), float(posterior), float(conf)

