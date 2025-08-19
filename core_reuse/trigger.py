# core_reuse/trigger.py
"""
Breakout trigger with three entry modes:
- baseline: Donchian prior-bar + ATR buffer (with compression gating)
- thrust_breakout: breakout must occur with statistical impulse on the bar
- retest_ignition: record breakout, require shallow, time-boxed retrace, then re-ignite

Keeps your multi-horizon TSMOM regime gating upstream.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# ---------- helpers ----------
def _atr_ema(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    alpha = 1.0 / max(1, n)
    return tr.ewm(alpha=alpha, adjust=False).mean()

def _ret_zscore(close: pd.Series, win: int) -> pd.Series:
    r = np.log(close / close.shift(1))
    mu = r.rolling(win, min_periods=win).mean()
    sd = r.rolling(win, min_periods=win).std(ddof=0)
    z = (r - mu) / sd.replace(0, np.nan)
    return z

def _body_dom(row: pd.Series) -> float:
    rng = float(row["high"] - row["low"])
    if rng <= 0.0:
        return 0.0
    return float((row["close"] - row["open"]) / rng)

def _bb_width(close: pd.Series, window: int) -> pd.Series:
    m = close.rolling(window, min_periods=window).mean()
    s = close.rolling(window, min_periods=window).std()
    upper = m + 2 * s
    lower = m - 2 * s
    width = (upper - lower) / m.abs().clip(lower=1e-9)
    return width

# ---------- trigger ----------
class BreakoutAfterCompression:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # state for retest mode (per instance; one symbol per instance in harness)
        self.state: Dict[str, Any] = {"phase": "idle", "anchor": None, "since": None, "dir": None}

    def _compression_gate(self, df_1m: pd.DataFrame) -> bool:
        c = self.cfg["strategy"]["trigger"]["compression"]
        g = self.cfg["strategy"]["trigger"]["gating"]
        if not g.get("require_recent_compression", True):
            return True
        if len(df_1m) < max(c["bb_window"], c["lookback_for_recent_squeeze"], c["min_squeeze_bars"]) + 2:
            return False
        bb = _bb_width(df_1m["close"], c["bb_window"])
        tail = bb.iloc[-c["lookback_for_recent_squeeze"]:]
        pct_rank = tail.rank(pct=True).iloc[-1]
        st_now = df_1m["close"].rolling(c["min_squeeze_bars"]).std().iloc[-1]
        st_ref = df_1m["close"].rolling(c["bb_window"]).std().iloc[-1]
        return bool((pct_rank <= 0.30) and (st_now < st_ref))

    def check(self, df_1m: pd.DataFrame, df_hist: pd.DataFrame, regime: str) -> Optional[Dict[str, Any]]:
        """
        Returns dict {'direction','level','reason'} or None.
        - df_1m : full 1m series up to 'now' (used for compression gate if enabled)
        - df_hist: same as df_1m; kept separate for clarity / future split
        - regime: "LONG" | "SHORT" | "FLAT"  (already computed upstream)
        """
        if regime not in ("LONG", "SHORT"):
            return None
        if len(df_hist) < 100:
            return None
        if not self._compression_gate(df_1m):
            return None

        st = self.cfg["strategy"]["trigger"]
        don_len = int(st["breakout"]["donchian_lookback"])
        buf_mult_base = float(st["breakout"]["buffer_atr_mult"])
        atr_win = int(self.cfg["risk"]["atr"]["window"])

        # Donchian using prior bar (no peek)
        prior = df_hist.iloc[:-1]
        if len(prior) < don_len:
            return None
        don_high = prior["high"].tail(don_len).max()
        don_low  = prior["low"].tail(don_len).min()

        # ATR buffer
        atr_ser = _atr_ema(df_hist[["high","low","close"]], n=atr_win)
        atr_val = float(atr_ser.iloc[-1])

        # Baseline levels (prior-bar Donchian + ATR buffer)
        base_long_level  = float(don_high + buf_mult_base * atr_val)
        base_short_level = float(don_low  - buf_mult_base * atr_val)

        # Entry mode branch
        ent = self.cfg.get("entry", {}) or {}
        mode = str(ent.get("mode", "baseline")).lower()

        # -------- baseline --------
        if mode == "baseline":
            last_close = float(df_hist["close"].iloc[-1])
            if regime == "LONG" and last_close >= base_long_level:
                return {"direction": "LONG", "level": base_long_level, "reason": "donchian_breakout"}
            if regime == "SHORT" and last_close <= base_short_level:
                return {"direction": "SHORT", "level": base_short_level, "reason": "donchian_breakout"}
            return None

        # -------- thrust_breakout --------
        if mode == "thrust_breakout":
            th = ent.get("thrust", {}) or {}
            z_win    = int(th.get("zscore_window", 20))
            z_k      = float(th.get("zscore_k", 1.5))
            min_body = float(th.get("min_body_dom", 0.6))
            buf_mult = float(th.get("buffer_atr_mult", 0.25))

            z = _ret_zscore(df_hist["close"], z_win)
            last = df_hist.iloc[-1]
            body = _body_dom(last)
            zval = float(z.iloc[-1])

            if regime == "LONG":
                lvl = float(don_high + buf_mult * atr_val)
                if last["close"] >= lvl and zval >= z_k and body >= min_body:
                    return {"direction":"LONG","level": lvl, "reason":"thrust_breakout"}
            else:
                lvl = float(don_low - buf_mult * atr_val)
                if last["close"] <= lvl and zval <= -z_k and (-body) >= min_body:
                    return {"direction":"SHORT","level": lvl, "reason":"thrust_breakout"}
            return None

        # -------- retest_ignition --------
        if mode == "retest_ignition":
            rt = ent.get("retest", {}) or {}
            rmin = float(rt.get("retrace_min_atr", 0.3))
            rmax = float(rt.get("retrace_max_atr", 0.6))
            timelimit = int(rt.get("time_limit_bars", 30))
            z_k = float(rt.get("confirm_z_k", 0.8))
            min_body = float(rt.get("min_body_dom", 0.5))
            look = int(rt.get("confirm_lookback", 10))
            buf_mult = float(rt.get("buffer_atr_mult", 0.25))

            S = self.state  # single-symbol instance in harness

            last = df_hist.iloc[-1]
            z = _ret_zscore(df_hist["close"], max(10, look))  # safe min
            zval = float(z.iloc[-1])
            body = _body_dom(last)

            # Phase 1: detect a fresh baseline breakout, arm waiting for shallow retest
            if S["phase"] == "idle":
                if regime == "LONG" and float(last["close"]) >= base_long_level:
                    S.update({"phase":"waiting_retest","anchor": base_long_level, "since":0, "dir":"LONG"})
                elif regime == "SHORT" and float(last["close"]) <= base_short_level:
                    S.update({"phase":"waiting_retest","anchor": base_short_level, "since":0, "dir":"SHORT"})
                return None

            # Phase 2: shallow, time-boxed pullback then ignition
            if S["phase"] == "waiting_retest":
                S["since"] = int(S.get("since") or 0) + 1
                if S["since"] > timelimit:
                    S.update({"phase":"idle", "anchor":None, "since":None, "dir":None})
                    return None

                if S["dir"] == "LONG":
                    # retrace depth from anchor in ATRs
                    retrace = max(0.0, S["anchor"] - float(last["low"]))
                    retrace_atr = retrace / max(1e-9, atr_val)
                    micro_hi = float(df_hist["high"].rolling(look).max().iloc[-2])
                    if rmin <= retrace_atr <= rmax and last["close"] > micro_hi and zval >= z_k and body >= min_body:
                        lvl = float(last["close"] + buf_mult * atr_val)
                        S.update({"phase":"idle", "anchor":None, "since":None, "dir":None})
                        return {"direction":"LONG","level": lvl, "reason":"retest_ignition"}
                else:
                    retrace = max(0.0, float(last["high"]) - S["anchor"])
                    retrace_atr = retrace / max(1e-9, atr_val)
                    micro_lo = float(df_hist["low"].rolling(look).min().iloc[-2])
                    if rmin <= retrace_atr <= rmax and last["close"] < micro_lo and zval <= -z_k and (-body) >= min_body:
                        lvl = float(last["close"] - buf_mult * atr_val)
                        S.update({"phase":"idle", "anchor":None, "since":None, "dir":None})
                        return {"direction":"SHORT","level": lvl, "reason":"retest_ignition"}
                return None

        # Fallback / unknown mode
        return None

