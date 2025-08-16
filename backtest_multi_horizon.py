#!/usr/bin/env python3
import os, argparse
from typing import Dict
import pandas as pd
from tqdm import tqdm

from backtester.engine import MultiHorizonEngine, StrategyConfig
from backtester.execution import ExecutionConfig
from data_loader import load_ticks_for_months, build_minute_bars


def load_config(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_symbol(symbol: str, cfg: Dict, results_dir: str, debug: bool):
    data_dir = cfg["data"]["data_dir"]
    months = cfg["data"]["months"]
    interval = cfg["data"]["resample_interval"]

    try:
        ticks = load_ticks_for_months(symbol, data_dir, months)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] {e}")
        return None

    bars = build_minute_bars(ticks, interval=interval)
    if bars.empty:
        print(f"[WARN] No bars for {symbol} in months {months}")
        return None

    scfg = StrategyConfig(
        momentum_windows = cfg["strategy"]["momentum_windows"],
        breakout_lookback= cfg["strategy"]["breakout_lookback"],
        vol_window       = cfg["strategy"]["vol_window"],
        target_vol_annual= cfg["strategy"]["target_vol_annual"],
        max_leverage     = cfg["strategy"]["max_leverage"],
    )
    ecfg = ExecutionConfig(
        taker_fee_bps=cfg["broker"]["taker_fee_bps"],
        slippage_bps=cfg["broker"]["slippage_bps"],
        latency_ms=cfg["broker"]["latency_ms"],
    )
    engine = MultiHorizonEngine(
        symbol=symbol,
        bars=bars,
        strat_cfg=scfg,
        exec_cfg=ecfg,
        initial_capital=cfg["broker"]["initial_capital"],
        results_dir=results_dir
    )
    engine.run(debug=debug)

    eq_path = os.path.join(results_dir, f"{symbol}_equity.csv")
    if os.path.exists(eq_path):
        df = pd.read_csv(eq_path)
        return df.iloc[-1]["equity"] if not df.empty else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--debug", action="store_true", help="Write per-bar diagnostics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.results_dir, exist_ok=True)

    equities = []
    for s in tqdm(cfg["data"]["symbols"], desc="Symbols"):
        last_eq = run_symbol(s, cfg, args.results_dir, debug=args.debug)
        equities.append({"symbol": s, "final_equity": last_eq})

    pd.DataFrame(equities).to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    print("✅ Backtest done. See results/")


if __name__ == "__main__":
    main()
