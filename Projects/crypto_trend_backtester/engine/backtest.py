from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from .regime import compute_regime
from .signals import build_signal_frame, pass_micro_alignment
from .risk import Position, price_to_r, update_stop_for_be, hit_stop


def _merge_frames(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    rg = compute_regime(df, cfg["regime"]["macro_window"], cfg["regime"]["macro_tmin"])
    sg = build_signal_frame(df, cfg)
    out = rg.join(sg[["atr", "d_hi", "d_lo", "kappa_micro_short", "kappa_micro_long", "accel", "long_breakout", "short_breakout"]], how="left")
    return out


def _first_nonzero_sign_change(prev_dir: int, cur_dir: int) -> bool:
    return (prev_dir != 0) and (cur_dir != 0) and (np.sign(prev_dir) != np.sign(cur_dir))


def run_backtest_symbol(
    df: pd.DataFrame,
    cfg: Dict,
    symbol: str,
    progress: bool = True,
    test_start: Optional[pd.Timestamp] = None,
    test_end: Optional[pd.Timestamp] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run the minimal backtest for a single symbol.
    If test_start/end provided, only open new positions inside [test_start, test_end] (inclusive).
    Positions are still managed over the entire df time range to avoid leakage.
    Returns (trades_list, stats_dict).
    """
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    dfm = _merge_frames(df, cfg)
    ent_next = bool(cfg["entry"]["enter_on_next_bar"])
    cooldown_bars = int(cfg["regime"]["cooldown_bars_after_flip"])
    be_r = float(cfg["risk"]["breakeven_r"])
    sl_mult = float(cfg["risk"]["sl_atr_mult"])
    sl_exact_neg1 = bool(cfg["risk"]["sl_exact_neg1"])

    blockers = {"flat_regime": 0, "cooldown": 0, "no_align": 0, "no_breakout": 0}

    trades: List[Dict] = []
    pos: Optional[Position] = None
    pending_side: Optional[int] = None  # for next-bar entry
    pending_idx: Optional[int] = None
    last_flip_idx: Optional[int] = None
    last_regime_dir: int = 0

    idx = dfm.index
    n = len(dfm)

    rng = range(1, n - 1)  # start at 1 so prev exists; stop at n-1 to allow next-bar entry fill
    if not ent_next:
        rng = range(1, n)  # unused path; kept for completeness

    iterator = tqdm(rng, desc=f"{symbol}", disable=(not progress))

    # Precompute eligible test mask
    in_test_mask = pd.Series(True, index=dfm.index)
    if test_start is not None and test_end is not None:
        in_test_mask = (dfm.index >= test_start) & (dfm.index <= test_end)

    for i in iterator:
        row = dfm.iloc[i]
        ts = idx[i]

        # track regime flips for cooldown
        cur_dir = int(row["regime_dir"])
        if _first_nonzero_sign_change(last_regime_dir, cur_dir):
            last_flip_idx = i
        if cur_dir != 0:
            last_regime_dir = cur_dir

        # fill pending entry at open[i] if any
        if ent_next and pending_side is not None and pending_idx is not None and i == pending_idx + 1:
            open_px = float(dfm["open"].iloc[i])
            atr_at_entry = float(dfm["atr"].iloc[pending_idx])  # ATR from decision bar (robust)
            r_denom = atr_at_entry * sl_mult
            if pending_side > 0:
                sl = open_px - r_denom
            else:
                sl = open_px + r_denom
            pos = Position(side=pending_side, entry=open_px, r_denom=r_denom, sl=sl, be_on=False)
            pending_side, pending_idx = None, None

        # manage open position: check BE and stop
        if pos is not None:
            bar_high = float(dfm["high"].iloc[i])
            bar_low = float(dfm["low"].iloc[i])
            if be_r is not None and np.isfinite(be_r) and be_r > 0:
                update_stop_for_be(pos, bar_high, bar_low, be_r)
            hit = hit_stop(pos, bar_high, bar_low, sl_exact_neg1=sl_exact_neg1)
            if hit is not None:
                exit_px, r = hit
                trades.append({
                    "side": "LONG" if pos.side > 0 else "SHORT",
                    "entry_ts": idx[i],  # record current bar ts for exit row (approx)
                    "exit_ts": idx[i],
                    "entry": pos.entry,
                    "exit": float(exit_px),
                    "R": float(r),
                })
                pos = None

        # If still have position or pending, skip entry logic
        if pos is not None or (pending_side is not None):
            continue

        # ENTRY LOGIC (gates), only count blockers when in test slice (for WF)
        if not in_test_mask.iloc[i]:
            continue

        # Gate 1: Regime
        if cur_dir == 0:
            blockers["flat_regime"] += 1
            continue

        # Gate 1b: Cooldown after flip
        if last_flip_idx is not None and (i - last_flip_idx) < cooldown_bars:
            blockers["cooldown"] += 1
            continue

        # Gate 2: Micro alignment
        if not pass_micro_alignment(row, cfg["micro"]):
            blockers["no_align"] += 1
            continue

        # Gate 3: Ignition (structure) via Donchian + ATR buffer
        do_long = (cur_dir > 0) and bool(row["long_breakout"])
        do_short = (cur_dir < 0) and bool(row["short_breakout"])
        if not (do_long or do_short):
            blockers["no_breakout"] += 1
            continue

        # Queue next-bar entry
        side = 1 if do_long else -1
        pending_side, pending_idx = side, i

    # End-of-run liquidation if still open
    if pos is not None:
        last_close = float(dfm["close"].iloc[-1])
        r = price_to_r(pos.side, pos.entry, last_close, pos.r_denom)
        trades.append({
            "side": "LONG" if pos.side > 0 else "SHORT",
            "entry_ts": idx[-1],
            "exit_ts": idx[-1],
            "entry": pos.entry,
            "exit": last_close,
            "R": float(r),
        })
        pos = None

    # Stats
    Rs = np.array([t["R"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = (Rs > 0).sum()
    stats = {
        "symbol": symbol,
        "trades": int(len(trades)),
        "win_rate": float(wins / len(trades)) if len(trades) > 0 else 0.0,
        "avg_R": float(Rs.mean()) if len(Rs) > 0 else 0.0,
        "sum_R": float(Rs.sum()) if len(Rs) > 0 else 0.0,
        "blockers": blockers,
    }
    return trades, stats
