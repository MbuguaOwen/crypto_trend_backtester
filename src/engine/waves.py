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

        self._ts_1m = self.df1m.index.view('i8')
        self._ts_ev = self.df_event.index.view('i8') if not self.df_event.empty else np.array([], dtype=np.int64)

    def _pad_idx_1m(self, ts_ns: int) -> int:
        i = np.searchsorted(self._ts_1m, ts_ns, side='right') - 1
        return i if i >= 0 else -1

    def _bfill_idx_1m(self, ts_ns: int) -> int:
        i = np.searchsorted(self._ts_1m, ts_ns, side='left')
        return i if i < self._ts_1m.size else -1

    def _pad_idx_ev(self, ts_ns: int) -> int:
        if self._ts_ev.size == 0:
            return -1
        j = np.searchsorted(self._ts_ev, ts_ns, side='right') - 1
        return j if j >= 0 else -1

    # --- PREWARM: walk event bars up to ts_end and record W2 candidates for quantiles
    def prewarm_until(self, ts_end: pd.Timestamp):
        dfe = self.df_event
        if dfe.empty or ts_end < dfe.index[0]:
            return 0
        j = self._pad_idx_ev(int(ts_end.value))
        if j == -1:
            return 0
        seen = 0
        step = max(1, len(dfe.iloc[:j + 1]) // 1000)
        for jj in range(0, j + 1, step):
            ev = dfe.iloc[max(0, jj - self.max_lookback_ev): jj + 1]
            if len(ev) < 10:
                continue
            i0 = self._bfill_idx_1m(int(ev.index[0].value))
            i1 = self._pad_idx_1m(int(ev.index[-1].value))
            if i0 == -1 or i1 == -1:
                continue
            if (i1 - i0 + 1) < self.min_cov_minutes:
                continue
            tr = (ev['high'] - ev['low']).abs()
            aw_min, aw_max = self.cfg['waves']['adaptive']['atr_window_range']
            atr_ev = tr.rolling(int((aw_min + aw_max) // 2), min_periods=int((aw_min + aw_max) // 2)).mean()
            if atr_ev.isna().all():
                continue
            am_min, am_max = self.cfg['waves']['adaptive']['atr_mult_range']
            piv = self._zigzag_atr(ev, atr_ev, (am_min + am_max) / 2.0)
            w = self._w1_w2_from_pivots(piv)
            if w is None:
                continue
            armed, score, post, conf = self._score_w2(ev, w, 0.62, 0.60, 0.60)
            if not armed:
                continue
            pos_end = np.searchsorted(ev.index.view('i8'), int(w['w1_end'][0].value), side='right') - 1
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
        j = self._pad_idx_ev(int(ts.value))
        if j == -1:
            return {'armed': False}
        ev = dfe.iloc[max(0, j - self.max_lookback_ev): j + 1]
        if len(ev) < 10:
            return {'armed': False}

        i0 = self._bfill_idx_1m(int(ev.index[0].value))
        i1 = self._pad_idx_1m(int(ts.value))
        if i0 == -1 or i1 == -1:
            return {'armed': False}
        if (i1 - i0 + 1) < self.min_cov_minutes:
            return {'armed': False}

        p: WaveParams = self.ac.waves_params(i_bar_1m)
        tr = (ev['high'] - ev['low']).abs()
        atr_ev = tr.rolling(p.atr_window, min_periods=p.atr_window).mean()
        if atr_ev.isna().all():
            return {'armed': False}

        piv = self._zigzag_atr(ev, atr_ev, p.atr_mult0)
        w = self._w1_w2_from_pivots(piv)
        if w is None:
            return {'armed': False}

        armed, score, post, conf = self._score_w2(ev, w, p.w2_end_arm, p.w2_post_min, p.min_conf)
        if not armed:
            return {'armed': False}

        pos_end = np.searchsorted(ev.index.view('i8'), int(w['w1_end'][0].value), side='right') - 1
        if pos_end == -1:
            return {'armed': False}
        age_bars = (len(ev) - 1) - pos_end
        if age_bars > p.max_age_bars:
            return {'armed': False}

        self.ac.record_w2(score, post, conf, age_bars)

        if w['dir'] == 'LONG':
            w2_high = w['w1_end'][1]
            w2_low = w['w2_end'][1]
        else:
            w2_low = w['w1_end'][1]
            w2_high = w['w2_end'][1]

        return {
            'armed': True,
            'dir': w['dir'],
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

