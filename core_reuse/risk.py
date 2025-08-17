"""Shim RiskManager matching the interface used by the harness.
Replace with your live risk.py to achieve exact parity.
"""
from typing import Tuple
import numpy as np
import pandas as pd

from .utils import bps_to_frac

def _truerange(h, l, c_prev):
    return max(h-l, abs(h-c_prev), abs(l-c_prev))

def _ema(x, alpha):
    out = []
    v = None
    for xi in x:
        v = xi if v is None else (alpha * xi + (1 - alpha) * v)
        out.append(v)
    return out

def compute_atr_df(df_hlc: pd.DataFrame, window: int) -> pd.Series:
    trs = []
    prev_close = None
    for i, row in df_hlc.iterrows():
        h, l, c = row['high'], row['low'], row['close']
        tr = (h - l) if prev_close is None else max(h-l, abs(h-prev_close), abs(l-prev_close))
        trs.append(tr)
        prev_close = c
    s = pd.Series(trs, index=df_hlc.index)
    return s.rolling(window, min_periods=window).mean()

class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def compute_atr(self, df_hlc: pd.DataFrame, window: int, smoothing='ema', ema_halflife_bars=10) -> float:
        """
        Robust ATR:
        - Uses True Range (TR) with previous close
        - Returns NaN if insufficient bars
        - EMA smoothing via halflife; fallback to SMA if requested
        """
        w = max(int(window or 0), 1)
        # Need at least w bars, but TR uses prev close, so require w+1 raw bars
        if len(df_hlc) < (w + 1):
            return float('nan')

        h = df_hlc['high'].to_numpy(dtype=float)
        l = df_hlc['low'].to_numpy(dtype=float)
        c = df_hlc['close'].to_numpy(dtype=float)

        # previous close (first TR uses same close to avoid NaN)
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]

        tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
        tr_ser = pd.Series(tr, index=df_hlc.index).tail(w)

        if smoothing == 'ema':
            # alpha from halflife
            alpha = 1.0 - np.exp(np.log(0.5) / max(1, int(ema_halflife_bars)))
            # Use the internal EMA helper but index **positionally**
            atr_list = _ema(tr_ser.values, alpha)
            return float(atr_list[-1])
        else:
            # simple SMA ATR
            return float(tr_ser.mean())

    def initial_levels(self, side: str, entry: float, atr_val: float, symcfg: dict) -> tuple:
        stops = self.cfg['risk']['stops']
        sl_mult = symcfg.get('stops', {}).get('initial_sl_atr_mult', stops.get('initial_sl_atr_mult_default', 2.0))
        tp_mult = stops['take_profit_atr_mult']
        if side.upper() == 'LONG':
            sl = entry - sl_mult * atr_val
            tp = entry + tp_mult * atr_val
        else:
            sl = entry + sl_mult * atr_val
            tp = entry - tp_mult * atr_val
        return (float(sl), float(tp))

    def position_size_units(self, equity_usd: float, entry: float, sl: float, taker_fee_bps: float, side: str, max_leverage: float) -> float:
        risk_cap_frac = self.cfg['risk']['vol_targeting']['sizing']['per_trade_risk_cap_pct_equity'] / 100.0
        fee_frac = bps_to_frac(taker_fee_bps)
        distance = abs(entry - sl)
        if distance <= 0:
            return 0.0
        # risk per unit ~ distance; add fee cushion
        risk_usd = equity_usd * risk_cap_frac
        qty = (risk_usd / (distance + fee_frac * entry))
        # leverage cap via notional
        notional = qty * entry
        max_notional = equity_usd * self.cfg['risk']['vol_targeting']['sizing']['max_leverage']
        if notional > max_notional:
            qty = max_notional / entry
        return max(0.0, float(qty))
