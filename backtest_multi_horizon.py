#!/usr/bin/env python3
import os, argparse
from typing import Dict
import pandas as pd
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from backtester.engine import MultiHorizonEngine, StrategyConfig
from backtester.execution import ExecutionConfig
from data_loader import load_ticks_for_months, build_minute_bars


def load_config(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _bars_cache_path(symbol, data_dir, months, interval, results_dir):
    h = hashlib.sha1(("|".join(months)).encode()).hexdigest()[:8]
    cache_dir = Path(results_dir) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{h}_{interval}.parquet"


def run_symbol(symbol: str, cfg: Dict, results_dir: str, debug: bool):
    data_dir = cfg["data"]["data_dir"]
    months = cfg["data"]["months"]
    interval = cfg["data"]["resample_interval"]

    print(f"[{symbol}] Loading ticks for {months}…", flush=True)
    try:
        ticks = load_ticks_for_months(symbol, data_dir, months)
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        return None
    except ValueError as e:
        print(f"[ERROR] {e}")
        return None

    cache_path = _bars_cache_path(symbol, data_dir, months, interval, results_dir)
    if cache_path.exists():
        print(f"[{symbol}] Using cached {interval} bars → {cache_path.name}", flush=True)
        bars = pd.read_parquet(cache_path)
    else:
        print(f"[{symbol}] Building {interval} bars…", flush=True)
        bars = build_minute_bars(ticks, interval=interval)
        if not bars.empty:
            bars.to_parquet(cache_path, index=False)
    if bars.empty:
        print(f"[WARN] No bars for {symbol} in months {months}")
        return None

    print(f"[{symbol}] Running simulation on {len(bars):,} bars…", flush=True)
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

    syms = cfg["data"]["symbols"]
    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(syms))) as ex:
        futs = {ex.submit(run_symbol, s, cfg, args.results_dir, args.debug): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                last_eq = fut.result()
            except Exception as e:
                print(f"[ERROR] {s}: {e}")
                last_eq = None
            results.append({"symbol": s, "final_equity": last_eq})

    pd.DataFrame(results).to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    print("✅ Backtest done. See results/")


if __name__ == "__main__":
    main()
