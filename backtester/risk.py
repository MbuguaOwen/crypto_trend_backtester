from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    side: int  # 1 for long, -1 for short
    qty: float
    entry_price: float
    sl: float
    tp: float
    be_moved: bool = False
    tsl_active: bool = False


def atr(bars: pd.DataFrame, period: int) -> pd.Series:
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().shift(1)


def initial_sl_tp(price: float, atr_val: float, side: int, cfg: Dict) -> (float, float):
    sl_mult = cfg.get("sl_mult", 2.0)
    tp_mult = cfg.get("tp_mult", 3.0)
    if side == 1:
        sl = price - sl_mult * atr_val
        tp = price + tp_mult * atr_val
    else:
        sl = price + sl_mult * atr_val
        tp = price - tp_mult * atr_val
    return sl, tp


def maybe_move_be(pos: Position, current_price: float, cfg: Dict) -> None:
    if pos.be_moved:
        return
    threshold = cfg.get("be_threshold", 0.55)
    dist = abs(pos.tp - pos.entry_price)
    if abs(current_price - pos.entry_price) >= threshold * dist:
        pos.sl = pos.entry_price
        pos.be_moved = True


def maybe_trail(pos: Position, atr_val: float, current_price: float, cfg: Dict) -> None:
    arm_th = cfg.get("tsl", {}).get("arm_threshold", 0.75)
    atr_mult = cfg.get("tsl", {}).get("atr_mult", 1.25)
    dist = abs(pos.tp - pos.entry_price)
    if not pos.tsl_active and abs(current_price - pos.entry_price) >= arm_th * dist:
        pos.tsl_active = True
    if pos.tsl_active:
        if pos.side == 1:
            pos.sl = max(pos.sl, current_price - atr_mult * atr_val)
        else:
            pos.sl = min(pos.sl, current_price + atr_mult * atr_val)


def check_exit(pos: Position, bar: pd.Series) -> Optional[float]:
    """Return exit price if stop/target hit within ``bar`` else ``None``."""

    high = bar["high"]
    low = bar["low"]
    if pos.side == 1:
        if low <= pos.sl:
            return pos.sl
        if high >= pos.tp:
            return pos.tp
    else:
        if high >= pos.sl:
            return pos.sl
        if low <= pos.tp:
            return pos.tp
    return None
