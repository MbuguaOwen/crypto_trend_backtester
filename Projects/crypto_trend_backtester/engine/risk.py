from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Position:
    """
    Minimal position container.
    side: +1 (LONG) or -1 (SHORT)
    entry: entry price
    r_denom: ATR_at_entry * sl_atr_mult (defines 1R)
    sl: stop-loss price
    be_on: whether breakeven stop is active
    """
    side: int
    entry: float
    r_denom: float
    sl: float
    be_on: bool = False


def price_to_r(side: int, entry: float, price: float, r_denom: float) -> float:
    """
    Convert a price P&L to R units: side * (price - entry) / r_denom
    """
    return side * (price - entry) / float(r_denom)


def update_stop_for_be(pos: Position, bar_high: float, bar_low: float, be_r: float) -> None:
    """
    Arm breakeven if unrealized R >= be_r for long (use bar_high) / short (use bar_low).
    """
    if pos.be_on:
        return
    if pos.side > 0:
        if price_to_r(pos.side, pos.entry, bar_high, pos.r_denom) >= be_r:
            pos.sl = pos.entry
            pos.be_on = True
    else:
        if price_to_r(pos.side, pos.entry, bar_low, pos.r_denom) >= be_r:
            pos.sl = pos.entry
            pos.be_on = True


def hit_stop(pos: Position, bar_high: float, bar_low: float, sl_exact_neg1: bool = True) -> Optional[Tuple[float, float]]:
    """
    Check if stop is hit within the bar (intrabar logic).
    Returns (exit_price, R) if stop was hit, else None.
    If sl_exact_neg1 is True, R is clamped to exactly -1.0 on stop.
    """
    if pos.side > 0:
        # Long: stop triggers if low <= sl
        if bar_low <= pos.sl:
            r = price_to_r(pos.side, pos.entry, pos.sl, pos.r_denom)
            if sl_exact_neg1:
                r = -1.0
            return pos.sl, r
    else:
        # Short: stop triggers if high >= sl (note sl is above entry)
        if bar_high >= pos.sl:
            r = price_to_r(pos.side, pos.entry, pos.sl, pos.r_denom)
            if sl_exact_neg1:
                r = -1.0
            return pos.sl, r
    return None
