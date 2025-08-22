
import os
import argparse
import concurrent.futures
import yaml
from tqdm import tqdm

# top-level (picklable) progress printer for single-worker mode
def progress_printer(symbol: str, done: int, total: int, bar_dict={}):
    if symbol not in bar_dict:
        bar_dict[symbol] = tqdm(total=total, desc=symbol, ncols=100, position=len(bar_dict))
    bar = bar_dict[symbol]
    bar.n = done
    bar.refresh()

def _worker(args):
    cfg, symbol, use_progress = args
    from src.engine.backtest import run_for_symbol
    hook = (lambda s, d, t: progress_printer(s, d, t)) if use_progress else None
    return run_for_symbol(cfg, symbol, progress_hook=hook)

def main():
    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--workers", type=int, default=0, help="Number of processes (0 = all cores)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    symbols = list(cfg.get('symbols', []))
    if not symbols:
        print("No symbols specified in config")
        return

    max_workers = os.cpu_count() or 1
    workers = max_workers if args.workers in (None, 0) else max(1, args.workers)

    summaries = []
    # show per-task progress when parallel; detailed per-bar progress when workers == 1
    use_progress = (workers == 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, (cfg, sym, use_progress)) for sym in symbols]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="ALL", ncols=100):
            summaries.append(fut.result())

    for s in summaries:
        print(s)

if __name__ == "__main__":
    main()
