# core/risk.py
from typing import Tuple

def be_threshold(entry: float, fee_buffer_r: float, risk_r: float, side: str) -> float:
    """
    Returns the BE clamp price with fee buffer in R units.
    """
    buffer_px = fee_buffer_r * risk_r  # price units
    if side == "LONG":
        return entry + buffer_px
    else:
        return entry - buffer_px

def make_initial_sl_tp(entry: float, atr: float, atr_mult_sl: float, atr_mult_tp: float, side: str) -> Tuple[float,float]:
    if side == "LONG":
        sl = entry - atr_mult_sl * atr
        tp = entry + atr_mult_tp * atr
    else:
        sl = entry + atr_mult_sl * atr
        tp = entry - atr_mult_tp * atr
    return sl, tp
