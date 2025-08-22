"""Time-series momentum regime engine.

This module implements a causal, multi–time-frame regime detector based on
simple time-series momentum (TSMOM).  The detector works off of 1 minute OHLCV
data and for each configured timeframe calculates the directional bias by
examining the sign of the difference between the most recent close and prior
closes.  A majority vote across timeframes produces the overall regime.

The configuration expected by :class:`TSMOMRegime` matches the block below::

    regime:
      timeframes:
        "1m":  { lookback_closes: 30 }
        "5m":  { lookback_closes: 20 }
        "15m": { lookback_closes: 12 }
        "1h":  { lookback_closes: 8 }
        "5h":  { lookback_closes: 3 }
      vote:
        require: 4

For each timeframe we resample the 1m data to the target bar size using a
right‑closed, label='right' rule so that the bar ending at ``ts`` only contains
data up to and including ``ts``.  The direction for a timeframe is determined
by summing the signs of ``close_t - close_{t-k}`` for ``k = 1 .. N`` where ``N``
is the ``lookback_closes``.  The score for a timeframe is the average of these
signs and acts as a conviction metric.  Optionally a ``strength`` metric is
computed from the average absolute z-score of the percentage returns used in
the sign calculation.

The overall regime direction is decided by a majority vote.  If the number of
bullish timeframes exceeds or equals ``require`` and strictly exceeds the
number of bearish timeframes the regime is ``BULL``; the converse gives ``BEAR``
and otherwise the regime is ``FLAT``.  The overall score is the mean of the
timeframe scores and the strength is the median of the per-timeframe strengths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TF_RULE = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1H",
    "5h": "5H",
}


def _resample_ohlcv(df1m: pd.DataFrame, tf_key: str) -> pd.DataFrame:
    """Resample 1m OHLCV to target timeframe, label/right-closed, causal."""
    rule = TF_RULE[tf_key]
    if rule == "1min":  # passthrough for 1m
        return df1m
    return (
        df1m.resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


class TSMOMRegime:
    """Multi-TF TSMOM regime with per-timeframe lookback and majority vote."""

    def __init__(self, cfg: dict, df1m: pd.DataFrame):
        self.cfg = cfg
        # Mapping timeframe -> {lookback_closes: int}
        self.specs = cfg["regime"]["timeframes"]
        self.require = int(cfg["regime"]["vote"]["require"])

        # Pre-resample once for efficiency.  The returned frames are causal and
        # we slice them using ``get_indexer`` when computing the regime.
        self.frames = {tf: _resample_ohlcv(df1m, tf) for tf in self.specs.keys()}

    # ------------------------------------------------------------------
    def _tf_direction_and_score(
        self, df_tf: pd.DataFrame, ts: pd.Timestamp, n_closes: int
    ) -> tuple[int, float, float] | None:
        """Return ``(dir_int, score, strength)`` for a single timeframe.

        ``dir_int`` is ``-1`` for bearish, ``0`` for tie and ``+1`` for bullish.
        ``score`` is the average of the individual signs (conviction in
        ``[-1, +1]``) and ``strength`` is the mean absolute z-score of the
        percentage returns used for the signs.  ``None`` is returned if there is
        insufficient history (warm-up guard).
        """

        # Guard: timestamp earlier than first bar
        if ts < df_tf.index[0]:
            return None

        # Get index position of the bar at or immediately preceding ``ts``
        j = df_tf.index.get_indexer([ts], method="pad")[0]
        if j == -1 or j < n_closes:
            # Not enough data yet (need N+1 bars including current)
            return None

        close = df_tf["close"].iloc[: j + 1]

        signs = []
        pct = []
        last = close.iloc[-1]
        for k in range(1, n_closes + 1):
            prev = close.iloc[-1 - k]
            signs.append(np.sign(last - prev))
            pct.append((last / prev) - 1.0)

        signs = np.array(signs, dtype=float)
        dir_int = int(np.sign(signs.sum()))
        tf_score = float(signs.mean())

        # Strength: mean absolute z-score of the percentage returns
        x = np.array(pct, dtype=float)
        if np.isfinite(x).all() and x.std() > 0:
            z = (x - x.mean()) / (x.std() + 1e-12)
            tf_strength = float(np.abs(z).mean())
        else:
            tf_strength = 0.0

        return dir_int, tf_score, tf_strength

    # ------------------------------------------------------------------
    def compute_at(self, ts: pd.Timestamp) -> dict:
        """Compute regime at timestamp ``ts``.

        Returns a dictionary with ``dir`` (``BULL``, ``BEAR`` or ``FLAT``),
        ``score`` and ``strength``.
        """

        votes = []
        tf_scores = []
        tf_strengths = []

        # Strict warm-up: if any TF doesn't have enough data -> FLAT
        for tf, spec in self.specs.items():
            n = int(spec.get("lookback_closes", 3))
            frame = self.frames[tf]
            res = self._tf_direction_and_score(frame, ts, n)
            if res is None:
                return {"dir": "FLAT", "score": 0.0, "strength": 0.0}

            dir_int, tf_score, tf_strength = res
            votes.append(dir_int)
            tf_scores.append(tf_score)
            tf_strengths.append(tf_strength)

        bulls = sum(1 for v in votes if v > 0)
        bears = sum(1 for v in votes if v < 0)

        if bulls >= self.require and bulls > bears:
            direction = "BULL"
        elif bears >= self.require and bears > bulls:
            direction = "BEAR"
        else:
            direction = "FLAT"

        regime_score = float(np.mean(tf_scores)) if tf_scores else 0.0
        regime_strength = float(np.median(tf_strengths)) if tf_strengths else 0.0

        return {
            "dir": direction,
            "score": regime_score,
            "strength": regime_strength,
        }


__all__ = ["TSMOMRegime"]

