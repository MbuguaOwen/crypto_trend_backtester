from dataclasses import dataclass
from typing import Optional

@dataclass
class SymbolConstraints:
    min_qty: float
    step_size: float
    min_notional_usd: float

def _floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return (int(x / step)) * step

def ensure_min_qty_like_live(price: float, qty: float, constraints: SymbolConstraints) -> float:
    """Mimic live's _ensure_min_qty logic using config-provided constraints.
    - Round down to step_size
    - Enforce min_qty
    - Enforce min_notional_usd based on current price
    Returns 0.0 if order would violate any constraint after rounding.
    """
    q = max(0.0, _floor_to_step(qty, constraints.step_size))
    if q < constraints.min_qty:
        return 0.0
    notional = price * q
    if notional < constraints.min_notional_usd:
        return 0.0
    return q
