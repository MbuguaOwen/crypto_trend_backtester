import numpy as np
import pandas as pd

TF_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h":   "1h",   # NOTE: lower-case 'h' to avoid deprecation
    "5h":   "5h",
}

def _resample(df1m, tf):
    if tf == "1min":
        return df1m
    rule = TF_MAP[tf].lower()
    return df1m.resample(rule, label='right', closed='right').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna()

class TSMOMRegime:
    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.cfg = cfg
        self.tf_specs = cfg['regime']['ts_mom']['timeframes']
        self.require = int(cfg['regime']['ts_mom']['require_majority'])
        # pre-resample each TF once
        self.frames = {spec['tf']: _resample(df1m, spec['tf']) for spec in self.tf_specs}

    def precompute_on_1m(self, df1m_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Precompute regime direction aligned to df1m_index.
        Returns int8 array: -1=BEAR, 0=FLAT, +1=BULL
        """
        out_dir = np.zeros(len(df1m_index), dtype=np.int8)
        frames = {spec['tf']: self.frames[spec['tf']] for spec in self.tf_specs}
        ptrs = {tf: 0 for tf in frames}

        for i, ts in enumerate(df1m_index):
            votes = []
            for spec in self.tf_specs:
                tf = spec['tf']; k = int(spec.get('lookback_closes', 3))
                f = frames[tf]
                p = ptrs[tf]
                fi = f.index
                # advance pointer
                while p + 1 < len(fi) and fi[p + 1] <= ts:
                    p += 1
                ptrs[tf] = p
                if p < k:
                    v = 0
                else:
                    close = f['close'].iloc[:p+1]
                    rets = [np.sign(close.iloc[-1] - close.shift(j).iloc[-1]) for j in range(1, k+1)]
                    v = int(np.sign(sum(rets)))
                votes.append(v)

            bulls = sum(1 for v in votes if v > 0)
            bears = sum(1 for v in votes if v < 0)
            if bulls >= self.require:
                out_dir[i] = 1
            elif bears >= self.require:
                out_dir[i] = -1
            else:
                out_dir[i] = 0

        return out_dir

    def compute_at(self, ts) -> dict:
        votes = []
        strength_parts = []

        for spec in self.tf_specs:
            tf = spec['tf']; k = int(spec.get('lookback_closes', 3))
            df = self.frames[tf]
            if ts < df.index[0]:
                return {'dir':'FLAT','score':0.0,'strength':0.0}
            idx = df.index.get_indexer([ts], method='pad')[0]
            if idx == -1:
                return {'dir':'FLAT','score':0.0,'strength':0.0}

            df_cut = df.iloc[:idx+1]
            if len(df_cut) < (k+1):
                return {'dir':'FLAT','score':0.0,'strength':0.0}

            close = df_cut['close']
            # signs of last k close-vs-close changes
            rets = [np.sign(close.iloc[-1] - close.shift(i).iloc[-1]) for i in range(1, k+1)]
            v = int(np.sign(sum(rets)))  # majority within TF; ties -> 0
            votes.append(v)

            # strength: mean |z| across those k closes
            pct = [ (close.iloc[-1] / close.shift(i).iloc[-1] - 1.0) for i in range(1, k+1) ]
            x = np.array(pct, dtype=float)
            if np.all(np.isfinite(x)) and x.std() > 0:
                z = (x - x.mean()) / (x.std() + 1e-12)
                strength_parts.append(np.abs(z).mean())

        bulls = sum(1 for v in votes if v > 0)
        bears = sum(1 for v in votes if v < 0)
        score = (bulls - bears) / max(1, len(self.tf_specs))

        dir_ = 'FLAT'
        if bulls >= self.require: dir_ = 'BULL'
        elif bears >= self.require: dir_ = 'BEAR'

        strength = float(np.median(strength_parts)) if strength_parts else 0.0
        return {'dir': dir_, 'score': float(score), 'strength': strength}
