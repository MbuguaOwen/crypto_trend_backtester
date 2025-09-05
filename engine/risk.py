# engine/risk.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskParams:
    sl_atr_mult: float = 15.0
    tp_atr_mult: float = 60.0
    breakeven_r: float | None = 0.5
    sl_exact_neg1: bool = True
    barrier_priority: str = "worst"  # "worst" | "best" | "tp_first" | "sl_first"

@dataclass
class Trade:
    symbol: str
    side: str           # "long" | "short"
    ts_entry: int
    entry: float
    atr: float
    sl: float
    tp: float
    be_armed: bool = False
    exit_reason: str | None = None
    ts_exit: int | None = None
    exit: float | None = None
    r_realized: float | None = None
    bars_held: int = 0

    def update_hold(self): self.bars_held += 1

def open_trade(symbol: str, side: str, ts_entry: int, entry: float, atr: float, rp: RiskParams) -> Trade:
    if side == "long":
        sl = entry - rp.sl_atr_mult * atr
        tp = entry + rp.tp_atr_mult * atr
    else:
        sl = entry + rp.sl_atr_mult * atr
        tp = entry - rp.tp_atr_mult * atr
    return Trade(symbol, side, ts_entry, entry, atr, sl, tp)

def _r_from_price(tr: Trade, px: float) -> float:
    if tr.side == "long":
        return (px - tr.entry) / (tr.atr * tr.__dict__.get("atr_mult", 1.0))
    else:
        return (tr.entry - px) / (tr.atr * tr.__dict__.get("atr_mult", 1.0))

def maybe_arm_be(tr: Trade, high: float, low: float, rp: RiskParams):
    if not rp.breakeven_r or tr.be_armed:
        return
    hit = False
    if tr.side == "long":
        hit = (high - tr.entry) >= rp.breakeven_r * tr.atr
    else:
        hit = (tr.entry - low) >= rp.breakeven_r * tr.atr
    if hit:
        tr.be_armed = True
        # Move SL to entry
        tr.sl = tr.entry

def _priority_order(rp: RiskParams):
    if rp.barrier_priority == "tp_first": return ("tp","sl")
    if rp.barrier_priority == "sl_first": return ("sl","tp")
    # worst/best: choose order based on which is economically worse/better this bar
    return ("worst",)

def check_exit(tr: Trade, ts_bar: int, o: float, h: float, l: float, c: float, rp: RiskParams) -> bool:
    """
    Bar-based fill policy:
    - Evaluate BE arming BEFORE barrier checks (so SL may be moved to entry on this bar).
    - If both SL and TP are inside [low, high] of this bar, use rp.barrier_priority.
      'worst' assumes adverse fill (SL for longs if both hit; SL for shorts if both hit).
      'best' assumes favorable fill (TP for longs/shorts).
    - Otherwise choose whichever barrier was touched.
    """
    # Compute touches
    sl_hit = (l <= tr.sl <= h) if tr.side == "long" else (l <= tr.sl <= h)
    tp_hit = (l <= tr.tp <= h) if tr.side == "long" else (l <= tr.tp <= h)

    if sl_hit and tp_hit:
        order = _priority_order(rp)
        if order[0] == "worst":
            take = "sl"
        elif order[0] == "best":
            take = "tp"
        else:
            take = order[0]
    elif sl_hit:
        take = "sl"
    elif tp_hit:
        take = "tp"
    else:
        return False

    tr.ts_exit = ts_bar
    tr.exit_reason = take
    if take == "sl":
        tr.exit = tr.sl
        tr.r_realized = -1.0 if rp.sl_exact_neg1 else _r_from_price(tr, tr.exit)
    else:
        tr.exit = tr.tp
        tr.r_realized = _r_from_price(tr, tr.exit)
    return True
