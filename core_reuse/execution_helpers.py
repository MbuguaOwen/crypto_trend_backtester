from __future__ import annotations
import math

def floor_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step

def ensure_min_qty(amount: float, price: float, *, amount_step: float, min_qty: float, min_notional: float):
    """
    Mirror CcxtExchange._ensure_min_qty semantics:
    - Round down to step.
    - Enforce >= min_qty and amount*price >= min_notional.
    Returns (rounded_amount, is_valid: bool).
    """
    amt = floor_step(float(amount), float(amount_step))
    if amt <= 0:
        return 0.0, False
    if amt < float(min_qty):
        return 0.0, False
    if amt * float(price) < float(min_notional):
        return 0.0, False
    return amt, True
