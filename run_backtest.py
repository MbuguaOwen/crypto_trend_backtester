import os
import argparse
import concurrent.futures
import yaml
from tqdm import tqdm


def progress_hook(total):
    pass


def _worker(args):
    cfg, symbol = args
    from src.engine.backtest import run_for_symbol
    return run_for_symbol(cfg, symbol, progress_hook=None)


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
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, (cfg, sym)) for sym in symbols]
        for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="ALL"):
            summaries.append(fut.result())

    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
