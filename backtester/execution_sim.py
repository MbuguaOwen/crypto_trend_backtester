from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def apply_slippage(price: float, atr_val: float, cfg: Dict, side: int) -> float:
    spread_bps = cfg.get("slippage", {}).get("min_spread_bps", 1.0)
    atr_div = cfg.get("slippage", {}).get("atr_divisor", 1000)
    impact = max(spread_bps / 10000 * price, atr_val / atr_div)
    return price + side * impact


def apply_fees(price: float, qty: float, cfg: Dict) -> float:
    fee_bps = cfg.get("taker_fee_bps", 4)
    return abs(price * qty) * fee_bps / 10000.0


def simulate_fill(price: float, qty: float, side: int, atr_val: float, cfg: Dict) -> Tuple[float, float]:
    price = apply_slippage(price, atr_val, cfg, side)
    fee = apply_fees(price, qty, cfg)
    return price, fee
