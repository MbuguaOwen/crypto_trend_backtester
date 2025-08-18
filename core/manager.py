# core/manager.py
import math
from typing import Dict, List
import pandas as pd
from .trade import Trade, EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL
from .risk import be_threshold, make_initial_sl_tp

def sign(side: str) -> int:
    return 1 if side == "LONG" else -1

def trail_candidate(tr: Trade, atr: float, atr_mult_tsl: float, from_which: str) -> float:
    if tr.side == "LONG":
        anchor = tr.highest_close if from_which == "highest_close" else tr.mfe_price
        return anchor - atr_mult_tsl * atr
    else:
        anchor = tr.lowest_close if from_which == "lowest_close" else tr.mfe_price
        return anchor + atr_mult_tsl * atr

def update_mfe_and_anchors(tr: Trade, o: float, h: float, l: float, c: float):
    if tr.side == "LONG":
        tr.mfe_price = max(tr.mfe_price or tr.entry, h)
        tr.highest_close = max(tr.highest_close or tr.entry, c)
    else:
        tr.mfe_price = min(tr.mfe_price or tr.entry, l)
        tr.lowest_close = min(tr.lowest_close or tr.entry, c)

def unrealized_r(tr: Trade, h: float, l: float) -> float:
    if tr.side == "LONG":
        upx = max(0.0, h - tr.entry)
    else:
        upx = max(0.0, tr.entry - l)
    return upx / max(1e-12, tr.risk_r_denom)

def maybe_activate_trail(tr: Trade, atr: float, cfg_tsl: dict):
    if tr.trail_active:
        return
    needed = cfg_tsl.get("trail_activation_r", 0.8)
    ur = tr.activation_r or 0.0
    if ur >= needed:
        tr.trail_active = True
        # clamp initial trail to profit side of entry
        from_which = cfg_tsl.get("trail_from", "highest_close")
        cand = trail_candidate(tr, atr, cfg_tsl.get("atr_mult_tsl", 3.0), from_which)
        fee_clamp = tr.entry + (tr.risk_r_denom * cfg_tsl.get("fee_buffer_r", 0.08)) if tr.side=="LONG" else tr.entry - (tr.risk_r_denom * cfg_tsl.get("fee_buffer_r", 0.08))
        if tr.side == "LONG":
            tr.trail_price = max(cand, fee_clamp)
        else:
            tr.trail_price = min(cand, fee_clamp)

def update_trail(tr: Trade, atr: float, cfg_tsl: dict):
    if not tr.trail_active:
        return
    from_which = cfg_tsl.get("trail_from", "highest_close")
    cand = trail_candidate(tr, atr, cfg_tsl.get("atr_mult_tsl", 3.0), from_which)
    fee_clamp_r = cfg_tsl.get("fee_buffer_r", 0.08)
    fee_clamp = tr.entry + (tr.risk_r_denom * fee_clamp_r) if tr.side=="LONG" else tr.entry - (tr.risk_r_denom * fee_clamp_r)
    if tr.side == "LONG":
        tr.trail_price = max(tr.trail_price or -math.inf, cand, fee_clamp)
    else:
        tr.trail_price = min(tr.trail_price or math.inf, cand, fee_clamp)

def check_exit(tr: Trade, o: float, h: float, l: float, c: float, cfg: dict, ts) -> bool:
    """
    Deterministic precedence per invariants:
    LONG: SL -> BE (only if trail not active) -> TSL -> TP
    SHORT: mirror order
    Price-touching uses intrabar H/L.
    """
    # precedence paths
    fee_buffer_r = cfg["risk"].get("fee_buffer_r", 0.08)
    if tr.side == "LONG":
        # SL
        if l <= tr.initial_sl:
            tr.exit_type = EXIT_SL
            tr.exit_ts = ts
            tr.exit_price = tr.initial_sl
            return True
        # BE (only if not trail active)
        be_px = tr.entry + (fee_buffer_r * tr.risk_r_denom)
        if (not tr.trail_active) and (l <= be_px):
            tr.exit_type = EXIT_BE
            tr.exit_ts = ts
            tr.exit_price = be_px
            return True
        # TSL
        if tr.trail_active and tr.trail_price is not None and l <= tr.trail_price:
            # invariant A guard (profit-protective clamp)
            clamp_px = tr.entry + (fee_buffer_r * tr.risk_r_denom)
            exit_px = max(tr.trail_price, clamp_px)
            tr.exit_type = EXIT_TSL
            tr.exit_ts = ts
            tr.exit_price = exit_px
            return True
        # TP
        if h >= tr.tp:
            tr.exit_type = EXIT_TP
            tr.exit_ts = ts
            tr.exit_price = tr.tp
            return True
    else:
        # SHORT mirror
        if h >= tr.initial_sl:
            tr.exit_type = EXIT_SL
            tr.exit_ts = ts
            tr.exit_price = tr.initial_sl
            return True
        be_px = tr.entry - (fee_buffer_r * tr.risk_r_denom)
        if (not tr.trail_active) and (h >= be_px):
            tr.exit_type = EXIT_BE
            tr.exit_ts = ts
            tr.exit_price = be_px
            return True
        if tr.trail_active and tr.trail_price is not None and h >= tr.trail_price:
            clamp_px = tr.entry - (fee_buffer_r * tr.risk_r_denom)
            exit_px = min(tr.trail_price, clamp_px)
            tr.exit_type = EXIT_TSL
            tr.exit_ts = ts
            tr.exit_price = exit_px
            return True
        if l <= tr.tp:
            tr.exit_type = EXIT_TP
            tr.exit_ts = ts
            tr.exit_price = tr.tp
            return True
    return False
