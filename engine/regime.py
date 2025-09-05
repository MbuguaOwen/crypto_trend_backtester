from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from .features import logret, rolling_tstat_of_mean


def compute_regime(df: pd.DataFrame, macro_window: int, macro_tmin: float) -> pd.DataFrame:
    """
    Compute macro regime via t-stat of mean 1m log returns.
    Adds columns:
      - r1m
      - kappa_macro
      - regime_dir: {+1: BULL, -1: BEAR, 0: FLAT}
      - regime_flip_i: cumulative count of flips between +1 and -1 (FLAT ignored)
    """
    out = df.copy()
    out["r1m"] = logret(out["close"])
    out["kappa_macro"] = rolling_tstat_of_mean(out["r1m"], macro_window)

    dir_series = pd.Series(0, index=out.index, dtype=int)
    dir_series = dir_series.mask(out["kappa_macro"] >= macro_tmin, 1)
    dir_series = dir_series.mask(out["kappa_macro"] <= -macro_tmin, -1)
    out["regime_dir"] = dir_series

    # Count flips in sign among non-zero states
    last_nonzero = 0
    flips = []
    flip_count = 0
    for v in out["regime_dir"].values:
        if v == 0:
            flips.append(flip_count)
        else:
            if last_nonzero != 0 and v != last_nonzero:
                flip_count += 1
            flips.append(flip_count)
            last_nonzero = v
    out["regime_flip_i"] = np.array(flips, dtype=int)
    return out
