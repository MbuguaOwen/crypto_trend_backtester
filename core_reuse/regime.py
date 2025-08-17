"""Shim for live regime.TSMOMRegime.

If you have your live module, drop it here or update the import below to point to your live package.
This shim provides a simple, *approximate* stand-in so the zip runs out of the box.
"""
from typing import Dict
import pandas as pd
import numpy as np

class TSMOMRegime:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _norm_mom(self, closes: pd.Series) -> float:
        if len(closes) < 3:
            return 0.0
        rets = np.diff(np.log(closes.values))
        if len(rets) == 0:
            return 0.0
        s = rets.sum()
        v = np.std(rets) if np.std(rets) > 1e-12 else 1.0
        return float(s / v)

    def classify(self, tf_data: Dict[str, pd.DataFrame]) -> str:
        cons = self.cfg['strategy']['tsmom_regime']['consensus']
        weights = cons['weights']
        lt = cons['long_threshold']
        st = cons['short_threshold']
        score = 0.0
        for tf, w in weights.items():
            df = tf_data.get(tf)
            if df is None or 'close' not in df.columns or len(df) == 0:
                continue
            lookback = self.cfg['strategy']['tsmom_regime']['timeframes'][tf]['lookback_closes']
            window = df['close'].tail(lookback)
            score += float(w) * self._norm_mom(window)
        if score >= lt:
            return "LONG"
        if score <= st:
            return "SHORT"
        return "FLAT"
