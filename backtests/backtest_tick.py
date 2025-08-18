# backtests/backtest_tick.py
"""
Tick→bar backtest with profit-protective TSL (invariants A/B/C), walk-forward safe.
- Signals on bar close; orders execute next bar open.
- Intrabar exits via H/L with precedence: SL→BE→TSL→TP (long), mirrored for shorts.
- Logs R-native metrics for proper analysis.
"""
import os, argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from core.config import load_config
from core.bar_builder import read_ticks_to_bars
from core.signals import compute_signals
from core.risk import make_initial_sl_tp, be_threshold
from core.trade import Trade, EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL
from core.manager import update_mfe_and_anchors, unrealized_r, maybe_activate_trail, update_trail, check_exit, sign

def _months_list(s: str):
    # "2025-01,2025-02" or "2025-07"
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]

def _load_bars_for_symbol(data_dir: str, symbol: str, months: list, bar_minutes: int) -> pd.DataFrame:
    dfs = []
    for m in months:
        fp = Path(data_dir) / f"{symbol}-ticks-{m}.csv"
        if not fp.exists():
            print(f"[WARN] Missing {fp}")
            continue
        dfs.append(read_ticks_to_bars(str(fp), bar_minutes=bar_minutes))
    if not dfs:
        raise FileNotFoundError("No monthly tick files found.")
    bars = pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
    return bars

def _place_entry(side: str, row: pd.Series, qty: float, cfg: dict) -> Trade:
    atr = row["atr"]
    entry = row["open"]  # next bar open
    sl, tp = make_initial_sl_tp(entry, atr, cfg["risk"]["atr_mult_sl"], cfg["risk"]["atr_mult_tp"], side)
    risk_r = abs(entry - sl)
    tr = Trade(
        symbol=row.get("symbol",""),
        side=side,
        entry_ts=row["ts"],
        entry=entry,
        qty=qty,
        initial_sl=sl,
        tp=tp,
        fee_bps=cfg["risk"]["fee_bps"],
        risk_r_denom=risk_r,
    )
    tr.mfe_price = entry
    tr.highest_close = entry
    tr.lowest_close  = entry
    return tr

def run_symbol(symbol: str, cfg: dict, data_dir: str, results_dir: str, months: list):
    bar_minutes = cfg["windows"].get("bar_minutes", 1)
    bars = _load_bars_for_symbol(data_dir, symbol, months, bar_minutes)
    bars["symbol"] = symbol

    # compute signals on CLOSE; entries fire next bar open
    sig = compute_signals(bars, cfg)
    df = sig.copy()

    # backtest state
    open_trade = None
    trades: list = []
    qty = 1.0

    # iterate bars; when signal at t, enter at t+1 open
    for i in tqdm(range(len(df)-1), desc=f"[{symbol}] running"):
        row = df.iloc[i]
        nxt = df.iloc[i+1]

        # Entry logic
        if open_trade is None:
            if row["long_signal"]:
                open_trade = _place_entry("LONG", nxt, qty, cfg)
                open_trade.symbol = symbol
                entry_fee = open_trade.fees(open_trade.entry)
            elif row["short_signal"]:
                open_trade = _place_entry("SHORT", nxt, qty, cfg)
                open_trade.symbol = symbol
                entry_fee = open_trade.fees(open_trade.entry)

        # Manage open trade
        if open_trade is not None:
            # update MFE and anchors using current bar (nxt)
            update_mfe_and_anchors(open_trade, nxt["open"], nxt["high"], nxt["low"], nxt["close"])

            # update activation_r
            ur = unrealized_r(open_trade, nxt["high"], nxt["low"])
            open_trade.activation_r = max(open_trade.activation_r or 0.0, ur)

            # maybe activate trail
            tsl_cfg = cfg.get("tsl", {})
            # inject fee_buffer_r from risk into tsl_cfg for manager functions
            tsl_cfg = {**tsl_cfg, "fee_buffer_r": cfg["risk"].get("fee_buffer_r", 0.08)}
            maybe_activate_trail(open_trade, nxt["atr"], tsl_cfg)
            update_trail(open_trade, nxt["atr"], tsl_cfg)

            # exit checks (deterministic precedence, intrabar H/L)
            if check_exit(open_trade, nxt["open"], nxt["high"], nxt["low"], nxt["close"], cfg, nxt["ts"]):
                # compute pnl (simple; per-side)
                exit_fee = open_trade.fees(open_trade.exit_price)
                gross = (open_trade.exit_price - open_trade.entry) * sign(open_trade.side) * open_trade.qty
                pnl = gross - entry_fee - exit_fee
                open_trade.pnl = pnl
                trades.append(open_trade)
                open_trade = None

    # Export trades CSV with R-native fields
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    trades_csv = cfg["io"]["trades_csv"].format(results_dir=results_dir, symbol=symbol)
    rows = []
    for t in trades:
        r_at_exit = (t.exit_price - t.entry)/t.risk_r_denom if t.side=="LONG" else (t.entry - t.exit_price)/t.risk_r_denom
        mfe_r = (t.mfe_price - t.entry)/t.risk_r_denom if t.side=="LONG" else (t.entry - t.mfe_price)/t.risk_r_denom
        rows.append({
            "ts_open": t.entry_ts,
            "ts_close": t.exit_ts,
            "symbol": symbol,
            "side": t.side,
            "entry": t.entry,
            "exit": t.exit_price,
            "qty": t.qty,
            "pnl": t.pnl,
            "exit_type": t.exit_type,
            "entry_reason": "breakout",
            "initial_sl": t.initial_sl,
            "tp": t.tp,
            "risk_r_denom": t.risk_r_denom,
            "mfe_price": t.mfe_price,
            "mfe_r": mfe_r,
            "r_at_exit": r_at_exit,
            "trail_active": t.trail_active,
            "trail_at_exit": t.trail_price,
            "trail_activation_r": t.activation_r,
        })
    df_tr = pd.DataFrame(rows)
    df_tr.to_csv(trades_csv, index=False)

    # summary
    counts = df_tr["exit_type"].value_counts().to_dict()
    net = float(df_tr["pnl"].sum()) if len(df_tr) else 0.0
    summary_csv = cfg["io"]["summary_csv"].format(results_dir=results_dir, symbol=symbol)
    sm = {
        "symbol": symbol,
        "trades": len(df_tr),
        "TP": int(counts.get("TP",0)),
        "TSL": int(counts.get("TSL",0)),
        "SL": int(counts.get("SL",0)),
        "BE": int(counts.get("BE",0)),
        "net_pnl": net
    }
    pd.DataFrame([sm]).to_csv(summary_csv, index=False)
    return sm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--months", required=True, help="Comma-separated YYYY-MM list")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dump-config", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    months = _months_list(args.months)

    if args.dump_config:
        dump_path = Path(args.results_dir)/"merged_config.json"
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        with open(dump_path,"w") as f:
            json.dump(cfg, f, indent=2, default=str)

    results = []
    for sym in args.pairs:
        res = run_symbol(sym, cfg, args.data_dir, args.results_dir, months)
        results.append(res)

    print("DONE:", results)

if __name__ == "__main__":
    main()
