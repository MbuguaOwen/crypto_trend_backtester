import argparse, os, sys
from collections import defaultdict
from tqdm import tqdm

from src.engine.backtest import run_all

def main():
    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    bars = {}
    def hook(symbol, done, total):
        if args.no_progress:
            return
        key = symbol.upper()
        if key not in bars:
            # Create a fresh tqdm per symbol
            bars[key] = tqdm(total=total, desc=f"{key} bars", position=len(bars), ncols=100, leave=True)
        # Clamp and update delta
        done = max(0, min(done, bars[key].total))
        delta = done - bars[key].n
        if delta > 0:
            bars[key].update(delta)

    # Run engine (this will also show internal per-symbol bar if enabled there)
    run_all(args.config, progress_hook=None if args.no_progress else hook)

    # Close bars cleanly
    for tb in bars.values():
        try:
            tb.close()
        except Exception:
            pass

if __name__ == "__main__":
    # Allow module execution: python -m run_backtest
    main()
