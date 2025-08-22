# src/engine/regime.py
import numpy as np
import pandas as pd

# Normalize pandas offsets (lower-case) to avoid deprecation chatter
TF_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h":   "1h",
    "5h":   "5h",
}

def _resample(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample 1m OHLCV to target tf with right-closed, right-labeled bars."""
    if tf == "1min":
        return df1m
    rule = TF_MAP[tf].lower()
    out = df1m.resample(rule, label='right', closed='right').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
        'volume':'sum',
    }).dropna()
    return out

class TSMOMRegime:
    """
    Majority-vote TSMOM regime with precomputed per-timeframe:
      - majority sign (-1 / 0 / +1) over last k closes
      - strength = mean(|z|) across last k pct changes

    NaN-safe, warmup-safe, and warning-free:
      * NaN contributions count as 0 in the vote
      * Strength rows with <2 finite values produce 0.0
      * No use of nanmean/nanstd → no runtime warnings
    """
    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.cfg = cfg
        self.tf_specs = cfg['regime']['ts_mom']['timeframes']
        self.require = int(cfg['regime']['ts_mom']['require_majority'])

        # Resample once per TF
        self.frames: dict[str, pd.DataFrame] = {
            spec['tf']: _resample(df1m, spec['tf']) for spec in self.tf_specs
        }

        # Precompute majority and strength series per TF
        self._prep: dict[str, dict[str, pd.Series]] = {}
        for spec in self.tf_specs:
            tf = spec['tf']
            k  = int(spec.get('lookback_closes', 3))
            df = self.frames[tf]
            close = df['close']

            # Build a (n, k) matrix of pct changes: (C_t / C_{t-i} - 1), i=1..k
            cols = [(close / close.shift(i) - 1.0) for i in range(1, k + 1)]
            pct = pd.concat(cols, axis=1)

            # ---------- Majority sign (NaN-safe) ----------
            # signs in {-1, 0, +1}; NaN -> 0 so it won't contaminate the sum
            x = pct.values.astype(float)                # (n, k)
            signs = np.sign(x)
            signs = np.where(np.isfinite(signs), signs, 0.0)  # NaN -> 0
            sums  = signs.sum(axis=1)
            maj   = np.sign(sums)
            maj   = np.where(np.isfinite(maj), maj, 0.0).astype(np.int8)
            maj_series = pd.Series(maj, index=df.index)

            # ---------- Strength = mean(|z|) over the row (NaN-safe, no warnings) ----------
            # We compute row-wise z using explicit masks and ddof=0 math, avoiding nanmean/nanstd.
            mask = np.isfinite(x)                       # finite entries
            count = mask.sum(axis=1, keepdims=True)     # (# finite in row)
            # row-wise mean over finite values; if none, mean=0
            sum_x = np.where(mask, x, 0.0).sum(axis=1, keepdims=True)
            safe_count = np.maximum(count, 1)           # avoid /0
            mu = sum_x / safe_count

            # centered only where finite
            centered = np.where(mask, x - mu, 0.0)
            # variance with ddof=0 over finite values
            var = (centered ** 2).sum(axis=1, keepdims=True) / safe_count
            sd = np.sqrt(var)
            denom = np.where(sd > 0.0, sd, 1.0)         # avoid /0; flat rows -> z=0

            z = centered / denom
            absz = np.abs(z)
            # mean(|z|) over finite entries only
            sum_absz = np.where(mask, absz, 0.0).sum(axis=1, keepdims=True)
            strength_row = (sum_absz / safe_count).ravel()
            # rows with <2 finite entries effectively get small strength; clamp exactly to 0 when no info
            strength_row = np.where(count.ravel() >= 2, strength_row, 0.0)

            strength_series = pd.Series(strength_row, index=df.index, dtype='float64')

            self._prep[tf] = {
                'maj': maj_series,
                'str': strength_series,
            }

    def compute_at(self, ts) -> dict:
        votes = []
        strengths = []

        for spec in self.tf_specs:
            tf = spec['tf']
            df = self.frames[tf]

            # Before first bar or no pad index → FLAT
            if ts < df.index[0]:
                return {'dir': 'FLAT', 'score': 0.0, 'strength': 0.0}
            idx = df.index.get_indexer([ts], method='pad')[0]
            if idx == -1:
                return {'dir': 'FLAT', 'score': 0.0, 'strength': 0.0}

            v = int(self._prep[tf]['maj'].iat[idx])    # -1, 0, +1 (no NaNs)
            s = float(self._prep[tf]['str'].iat[idx])  # >= 0.0
            votes.append(v)
            strengths.append(s)

        bulls = sum(1 for v in votes if v > 0)
        bears = sum(1 for v in votes if v < 0)
        score = (bulls - bears) / max(1, len(self.tf_specs))

        dir_ = 'FLAT'
        if bulls >= self.require:
            dir_ = 'BULL'
        elif bears >= self.require:
            dir_ = 'BEAR'

        strength = float(np.median(strengths)) if strengths else 0.0
        return {'dir': dir_, 'score': float(score), 'strength': strength}
