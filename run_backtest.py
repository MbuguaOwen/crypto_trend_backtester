import os
import argparse
import warnings
import time
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import yaml
from tqdm import tqdm

# Import the existing single-symbol runner
from src.engine.backtest import run_for_symbol


def _worker(symbol: str, cfg_path: str, q=None):
    """Child process entry point.

    Each worker loads the YAML config, runs a single symbol via
    ``run_for_symbol`` and pushes progress updates to the queue ``q``.
    The queue carries simple tuples that are easily pickled across
    processes, keeping the worker Windows-safe.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    def hook(sym, done, total):
        if q is not None:
            q.put(("PROGRESS", sym, int(done), int(total)))

    summary = run_for_symbol(cfg, symbol, progress_hook=hook)
    if q is not None:
        q.put(("DONE", symbol, summary))
    return summary


def main():
    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester (parallel)")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of processes (0 = use all CPU cores)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--suppress-warnings", action="store_true", help="Hide warnings")
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    symbols = list(cfg["symbols"])
    if not symbols:
        print("No symbols found in config.")
        return

    max_procs = os.cpu_count() or 1
    workers = max_procs if args.workers in (None, 0) else max(1, args.workers)

    manager = mp.Manager()
    q = manager.Queue() if not args.no_progress else None
    bars = {}
    totals = {}
    overall = None
    last_done = defaultdict(int)

    def _drain_progress(non_block=False):
        nonlocal overall
        polled = 0
        while True:
            try:
                evt = q.get_nowait() if non_block else q.get(timeout=0.2)
            except Exception:
                break
            polled += 1
            kind = evt[0]
            if kind == "PROGRESS":
                _, sym, done, total = evt
                if sym not in bars and not args.no_progress:
                    bars[sym] = tqdm(total=total, desc=f"{sym} bars", ncols=100,
                                     mininterval=0.2, smoothing=0.1,
                                     leave=True, dynamic_ncols=True,
                                     position=len(bars))
                    totals[sym] = total
                if not args.no_progress:
                    incr = max(0, done - last_done[sym])
                    if incr:
                        bars[sym].update(incr)
                        last_done[sym] = done

                    if overall is None:
                        tot_all = sum(totals.values()) if totals else 0
                        if tot_all > 0:
                            pos = len(bars) + 1
                            overall = tqdm(total=tot_all, desc="ALL bars",
                                           ncols=100, mininterval=0.2, smoothing=0.1,
                                           leave=True, dynamic_ncols=True,
                                           position=pos)
                    if overall is not None:
                        agg_done = sum(last_done[s] for s in last_done.keys())
                        delta_overall = max(0, agg_done - overall.n)
                        if delta_overall:
                            overall.update(delta_overall)
            elif kind == "DONE":
                pass
        return polled

    futures = []
    summaries = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for sym in symbols:
            fut = ex.submit(_worker, sym, args.config, q)
            futures.append(fut)

        running = True
        while running:
            running = any(not f.done() for f in futures)
            if q is not None:
                _drain_progress(non_block=True)
            time.sleep(0.05)

        for fut in as_completed(futures):
            s = fut.result()
            sym = s.get("symbol", "UNKNOWN")
            summaries[sym] = s

    if not args.no_progress:
        for tb in bars.values():
            try:
                tb.close()
            except Exception:
                pass
        if overall is not None:
            try:
                overall.close()
            except Exception:
                pass

    header = ["Symbol", "Trades", "Win%", "Avg R", "Sum R", "PSL", "TSL", "BE", "TP", "EOD"]
    rows = []
    totals_row = Counter()
    t_trades = 0
    t_win = 0.0
    t_avgR_num = 0.0
    t_avgR_den = 0

    for sym in symbols:
        s = summaries.get(sym, {})
        exits = (s.get("exits") or {}) if s else {}
        n_tr = int(s.get("trades", 0) or 0)
        win = float(s.get("win_rate", 0.0) or 0.0)
        avgR = float(s.get("avg_R", 0.0) or 0.0)
        sumR = float(s.get("sum_R", 0.0) or 0.0)

        PSL = int(exits.get("SL", 0))
        TSL = int(exits.get("TSL", 0))
        BE = int(exits.get("BE", 0))
        TP = int(exits.get("TP", 0))
        EOD = int(exits.get("EOD", 0))

        rows.append([sym, n_tr, f"{win*100:5.1f}%", f"{avgR:6.3f}", f"{sumR:7.2f}", PSL, TSL, BE, TP, EOD])

        totals_row["PSL"] += PSL
        totals_row["TSL"] += TSL
        totals_row["BE"] += BE
        totals_row["TP"] += TP
        totals_row["EOD"] += EOD
        t_trades += n_tr
        t_win += win * n_tr
        t_avgR_num += avgR * n_tr
        t_avgR_den += n_tr

    if t_avgR_den:
        t_win_rate = t_win / t_avgR_den
        t_avgR = t_avgR_num / t_avgR_den
    else:
        t_win_rate = 0.0
        t_avgR = 0.0

    print("\n=== Backtest Summary (per symbol) ===")
    print("{:<8} {:>6} {:>6} {:>7} {:>8} {:>5} {:>5} {:>5} {:>5} {:>5}".format(*header))
    for r in rows:
        print("{:<8} {:>6} {:>6} {:>7} {:>8} {:>5} {:>5} {:>5} {:>5} {:>5}".format(*r))
    print("\n--- TOTALS ---")
    print("Trades: {:d} | Win%: {:5.1f}% | Avg R: {:6.3f}".format(t_trades, t_win_rate*100, t_avgR))
    print("PSL: {:d} | TSL: {:d} | BE: {:d} | TP: {:d} | EOD: {:d}".format(
        totals_row["PSL"], totals_row["TSL"], totals_row["BE"], totals_row["TP"], totals_row["EOD"]
    ))


if __name__ == "__main__":
    mp.freeze_support()
    main()

