# engine/backtest.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

from . import signals
from .risk import RiskParams, open_trade, maybe_arm_be, check_exit


def run_insample(df: pd.DataFrame, symbol: str, cfg: dict, outdir: Path):
    df = signals.compute_features(df.copy(), cfg)
    rp = RiskParams(**cfg["risk"])

    trades = []
    tr = None
    flat_cd = 0
    warmup = max(
        cfg["features"]["atr_window"],
        cfg["regime"]["macro_window"],
        cfg["micro"]["long_window"],
        cfg["features"]["donchian_lookback"],
        cfg["filters"]["atr_pctile_window"],
    ) + 2

    for i in tqdm(range(warmup, len(df)), desc=f"{symbol}", leave=False):
        row_prev = df.iloc[i - 1]
        row = df.iloc[i]
        ts = int(row["timestamp"])

        # manage open position
        if tr is not None:
            maybe_arm_be(tr, row["high"], row["low"], rp)
            if check_exit(tr, ts, row["open"], row["high"], row["low"], row["close"], rp):
                trades.append(tr.__dict__)
                tr = None
                flat_cd = int(cfg["entry"].get("flat_cooldown_bars", 0))
            else:
                tr.update_hold()
                continue  # don’t look for new entries if we’re in a trade

        # flat cooldown
        if flat_cd > 0:
            flat_cd -= 1
            continue

        # new entry (next-bar open; ATR from prev bar)
        side = signals.entry_signal(row_prev, cfg)
        if side:
            atr = float(row_prev["atr"])
            if pd.notna(atr) and atr > 0:
                tr = open_trade(symbol, side, ts, float(row["open"]), atr, rp)

    # Force close remaining
    if tr is not None:
        tr.exit_reason = "eod"
        tr.ts_exit = int(df.iloc[-1]["timestamp"])
        tr.exit = float(df.iloc[-1]["close"])
        if tr.side == "long":
            tr.r_realized = (tr.exit - tr.entry) / (tr.atr)
        else:
            tr.r_realized = (tr.entry - tr.exit) / (tr.atr)
        trades.append(tr.__dict__)

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(outdir / "trades.csv", index=False)

    r = pd.Series([t["r_realized"] for t in trades if t["r_realized"] is not None])
    stats = {
        "symbol": symbol,
        "trades": int(len(trades)),
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "sum_R": float(r.sum()) if len(r) else 0.0,
        "avg_R": float(r.mean()) if len(r) else 0.0,
    }
    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats

