# run_backtest.py

import os
import argparse
import concurrent.futures
import queue
import time
import json
import yaml
from dataclasses import dataclass
from multiprocessing import Manager, get_start_method
from tqdm import tqdm


# ---------- Progress plumbing (top-level & picklable) ----------

@dataclass
class ProgressMsg:
    symbol: str
    done: int
    total: int


class ProgressSender:
    """Picklable callable: worker-safe hook to send progress to parent via a Manager.Queue."""
    def __init__(self, q):
        self.q = q
    def __call__(self, symbol: str, done: int, total: int):
        self.q.put(ProgressMsg(symbol, done, total))


def _worker(args):
    """
    Worker entrypoint (must be top-level to be pickled on Windows).
    Receives cfg (dict), symbol (str), and progress Queue (or None).
    """
    cfg, symbol, q = args
    from src.engine.backtest import run_for_symbol
    hook = ProgressSender(q) if q is not None else None
    return run_for_symbol(cfg, symbol, progress_hook=hook)


# ---------- Parent-side progress renderer ----------

class ProgressRenderer:
    """
    Parent-side renderer that owns tqdm bars per symbol.
    It consumes ProgressMsg items from a Queue and updates bars.
    """
    def __init__(self):
        self.bars = {}   # symbol -> tqdm instance
        self.totals = {} # symbol -> total bars

    def update_from_queue(self, q, timeout=0.05):
        """Drain queue (non-blocking-ish) and render updates."""
        processed = 0
        while True:
            try:
                msg = q.get(timeout=timeout if processed == 0 else 0.0)
            except queue.Empty:
                break
            if msg.symbol not in self.bars:
                # Create a new bar for this symbol
                self.totals[msg.symbol] = msg.total
                self.bars[msg.symbol] = tqdm(
                    total=msg.total,
                    desc=msg.symbol,
                    ncols=100,
                    position=len(self.bars),
                    leave=True
                )
            bar = self.bars[msg.symbol]
            # Update bar to absolute position (not incremental)
            bar.n = min(msg.done, bar.total)
            bar.refresh()
            processed += 1
        return processed

    def close_all(self):
        for bar in self.bars.values():
            try:
                bar.close()
            except Exception:
                pass


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

    # Shared queue for progress (even in single-worker mode so output is consistent)
    with Manager() as mgr:
        q = mgr.Queue()

        summaries = []
        renderer = ProgressRenderer()

        # Outer pool: keep the simple "ALL" bar to show task completion count
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, (cfg, sym, q)) for sym in symbols]
            remaining = set(futs)

            # ALL bar shows completed worker tasks (not per-bar minutes)
            all_bar = tqdm(total=len(futs), desc="ALL", ncols=100, position=0, leave=True)

            # Event loop: poll for progress & task completion
            while remaining:
                # Render any progress updates from workers
                renderer.update_from_queue(q, timeout=0.05)

                # Check which futures have completed since last pass
                done_now, remaining = concurrent.futures.wait(
                    remaining, timeout=0.10, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done_now:
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {"symbol": "UNKNOWN", "error": str(e)}
                    summaries.append(res)
                    all_bar.update(1)

                # Be nice to the event loop
                time.sleep(0.02)

            # Final drain to ensure bars reach 100%
            for _ in range(5):
                if renderer.update_from_queue(q, timeout=0.05) == 0:
                    break

            all_bar.close()
            renderer.close_all()

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
