import time
from typing import Optional


def utc_ms_now() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


def pct(a: float, b: float) -> Optional[float]:
    """Return percent change from a to b."""
    if a == 0:
        return None
    return (b - a) / a * 100.0


def ensure_min_qty(qty: float, step: float, min_amount: float, min_cost: float, last_price: float) -> float:
    """Adjust quantity to satisfy exchange min qty/step/cost constraints."""
    q = float(qty)
    if min_amount:
        q = max(q, float(min_amount))
    if min_cost and last_price:
        q = max(q, float(min_cost) / float(last_price))
    if step and step > 0:
        steps = (q / step)
        if steps != int(steps):
            from math import ceil
            q = ceil(steps) * step
    return float(q)
