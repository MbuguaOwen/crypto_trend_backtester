from __future__ import annotations
import numpy as np
import pandas as pd

class TSMOMRegime:
    """Reference implementation (used if live import is not available).
    Classifies BULL/BEAR/FLAT via multi-horizon lookback sign consensus.
    """
    def __init__(self, cfg: dict):
        s = cfg.get("strategy",{}).get("ts_mom",{})
        self.lookbacks = s.get("lookbacks",[20,60,120])
        self.consensus_threshold = float(s.get("consensus_threshold", 0.5))

    def classify(self, closes_by_tf: dict[str, pd.Series]) -> str:
        assert len(closes_by_tf)>0
        # choose highest frequency (has most points); use the one with max length
        key = max(closes_by_tf.keys(), key=lambda k: len(closes_by_tf[k]))
        c = closes_by_tf[key]
        votes = []
        for lb in self.lookbacks:
            r = c.pct_change(lb)
            votes.append(np.sign(r))
        sig = pd.concat(votes, axis=1).sum(axis=1)
        s_last = sig.shift(1).iloc[-1]
        frac = s_last / max(1, len(self.lookbacks))
        if frac > self.consensus_threshold:
            return "BULL"
        if frac < -self.consensus_threshold:
            return "BEAR"
        return "FLAT"
