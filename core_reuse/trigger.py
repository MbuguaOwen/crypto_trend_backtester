from __future__ import annotations
import pandas as pd
from .utils import bb_width, pct_rank

class BreakoutAfterCompression:
    """Donchian breakout conditioned on compression; prior-bar levels + ATR buffer.
    Returns tuple (should_enter: bool, side: str|None, reason: str|None).
    """
    def __init__(self, cfg: dict):
        t = cfg.get("strategy",{}).get("trigger",{})
        self.N = int(t.get("donchian_lookback", 20))
        comp = t.get("compression", {})
        self.bb_window = int(comp.get("bb_window", 50))
        self.width_pct_max = float(comp.get("width_percentile_max", 30))
        ks = t.get("ksigma",{})
        self.ks_enabled = bool(ks.get("enabled", False))
        self.ks_k = float(ks.get("k",2.0))
        self.ks_window = int(ks.get("window",60))
        self.atr_buffer_mult = float(t.get("atr_buffer_mult",0.0))

    def _donch_levels(self, df: pd.DataFrame):
        hh = df['high'].rolling(self.N, min_periods=self.N).max()
        ll = df['low'].rolling(self.N, min_periods=self.N).min()
        return hh, ll

    def _compression_gate(self, close: pd.Series) -> bool:
        w = bb_width(close, self.bb_window)
        pr = pct_rank(w, self.bb_window) * 100.0
        ok = (pr.shift(1).iloc[-1] <= self.width_pct_max)
        return bool(ok)

    def _ksigma(self, close: pd.Series):
        m = close.rolling(self.ks_window, min_periods=self.ks_window).mean()
        s = close.rolling(self.ks_window, min_periods=self.ks_window).std(ddof=0)
        up = (close.iloc[-1] > (m.shift(1).iloc[-1] + self.ks_k * s.shift(1).iloc[-1]))
        dn = (close.iloc[-1] < (m.shift(1).iloc[-1] - self.ks_k * s.shift(1).iloc[-1]))
        if up: return True, "ksigma_breakout_long"
        if dn: return True, "ksigma_breakout_short"
        return False, None

    def check(self, df_1m: pd.DataFrame, atr_last: float, regime: str):
        if len(df_1m) < max(self.N, self.bb_window, self.ks_window)+2:
            return False, None, None
        hh, ll = self._donch_levels(df_1m)
        # prior-bar levels + ATR buffer
        donch_h = hh.shift(1).iloc[-1]
        donch_l = ll.shift(1).iloc[-1]
        buf = (self.atr_buffer_mult * float(atr_last)) if pd.notna(atr_last) else 0.0
        hi = df_1m["high"].iloc[-1]
        lo = df_1m["low"].iloc[-1]

        # Compression gate
        gated = self._compression_gate(df_1m["close"])

        # Donchian breakout
        if gated and regime=="BULL" and hi > (donch_h + buf):
            return True, "LONG", "donchian_breakout"
        if gated and regime=="BEAR" and lo < (donch_l - buf):
            return True, "SHORT", "donchian_breakout"

        # Optional k-sigma fallback
        if self.ks_enabled:
            ok, why = self._ksigma(df_1m["close"])
            if ok and regime=="BULL" and "long" in why:
                return True, "LONG", "ksigma_breakout"
            if ok and regime=="BEAR" and "short" in why:
                return True, "SHORT", "ksigma_breakout"

        return False, None, None
