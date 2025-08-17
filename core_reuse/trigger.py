"""Shim for BreakoutAfterCompression with Donchian + ATR buffer and a compression gate.
Replace with your live trigger.py for parity.
"""
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

def _bb_width(close: pd.Series, window: int) -> pd.Series:
    m = close.rolling(window, min_periods=window).mean()
    s = close.rolling(window, min_periods=window).std()
    upper = m + 2*s
    lower = m - 2*s
    width = (upper - lower) / m.abs().clip(lower=1e-9)
    return width

class BreakoutAfterCompression:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def check(self, df_1m: pd.DataFrame, regime: str) -> Optional[Dict[str, Any]]:
        if len(df_1m) < 2:
            return None
        c = self.cfg['strategy']['trigger']['compression']
        g = self.cfg['strategy']['trigger']['gating']
        b = self.cfg['strategy']['trigger']['breakout']
        lookback = b.get('donchian_lookback', 25)
        buf_mult = b.get('buffer_atr_mult', 0.25)

        # compression gate
        if g.get('require_recent_compression', True):
            bb = _bb_width(df_1m['close'], c['bb_window'])
            bw = bb.iloc[-c['lookback_for_recent_squeeze']:]
            squeeze = (bw.rank(pct=True).iloc[-1] <= 0.3) and (df_1m['close'].rolling(c['min_squeeze_bars']).std().iloc[-1] < df_1m['close'].rolling(c['bb_window']).std().iloc[-1])
            if not bool(squeeze):
                return None

        # Donchian using prior bar
        prior = df_1m.iloc[:-1]
        if len(prior) < lookback:
            return None
        don_high = prior['high'].tail(lookback).max()
        don_low  = prior['low'].tail(lookback).min()

        # ATR buffer (use simple high-low)
        hl = prior[['high','low','close']].copy()
        tr = (hl['high'] - hl['low']).rolling(14, min_periods=1).mean().iloc[-1]
        buf = buf_mult * float(tr)

        last_close = df_1m['close'].iloc[-1]

        if regime == "LONG":
            level = don_high + buf
            if last_close > level:
                return {'direction': 'long', 'level': float(level), 'reason': 'donchian_breakout'}
        elif regime == "SHORT":
            level = don_low - buf
            if last_close < level:
                return {'direction': 'short', 'level': float(level), 'reason': 'donchian_breakout'}
        return None
