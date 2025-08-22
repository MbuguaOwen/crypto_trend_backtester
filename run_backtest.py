# run_backtest.py

import os
import argparse
import concurrent.futures
import json
import yaml
from tqdm import tqdm


# top-level (picklable) progress printer for single-worker mode
def progress_printer(symbol: str, done: int, total: int, bar_dict={}):
    if symbol not in bar_dict:
        # leave=True so bars persist after completion
        bar_dict[symbol] = tqdm(total=total, desc=symbol, ncols=100, position=len(bar_dict), leave=True)
    bar = bar_dict[symbol]
    bar.n = done
    bar.refresh()


def _worker(args):
    cfg, symbol, use_progress = args
    from src.engine.backtest import run_for_symbol
    # Only create a local hook in the child when single-worker; never pickle it
    hook = (lambda s, d, t: progress_printer(s, d, t)) if use_progress else None
    return run_for_symbol(cfg, symbol, progress_hook=hook)


def main():
    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--workers", type=int, default=0, help="Number of processes (0 = all cores)")
    args = ap.parse_args()

    # Load config once in parent
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    symbols = list(cfg.get("symbols", []))
    if not symbols:
        print("No symbols specified in config")
        return

    max_workers = os.cpu_count() or 1
    workers = max_workers if args.workers in (None, 0) else max(1, args.workers)

    use_progress = bool(cfg['logging']['progress']) and workers == 1

    summaries = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_worker, (cfg, sym, use_progress)) for sym in symbols]
        all_bar = tqdm(total=len(futs), desc="ALL", ncols=100, position=0, leave=True)
        for fut in concurrent.futures.as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                res = {"symbol": "UNKNOWN", "error": str(e)}
            summaries.append(res)
            all_bar.update(1)
        all_bar.close()

    # Print per-symbol summaries (also written by backtest.py individually)
    for s in summaries:
        print(s)

    # Write combined summary alongside other outputs
    out_dir = cfg["paths"]["outputs_dir"]
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, "combined_summary.json")
    with open(combined_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nCombined summary written to: {combined_path}")


if __name__ == "__main__":
    # Windows uses 'spawn': ensure all top-levels are picklable (they are).
    # This also works on POSIX.
    main()
