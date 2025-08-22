# run_backtest.py
import os
import argparse
import concurrent.futures
import multiprocessing as mp
import queue as pyqueue
import threading
import time
from typing import Any, Dict, List
import yaml
from tqdm import tqdm

# ------ Progress message structure ------
# {'type': 'init', 'symbol': str, 'total': int}
# {'type': 'tick', 'symbol': str, 'done': int}
# {'type': 'done', 'symbol': str}
# {'type': 'log',  'msg': str}


def _progress_listener(q: Any, stop_evt: threading.Event) -> None:
    """
    Runs in the MAIN process. Consumes progress messages from workers and updates tqdm bars.
    'q' is a Manager().Queue() or mp.Queue(); we keep it untyped (Any) for Windows compatibility.
    """
    bars: Dict[str, tqdm] = {}
    order: List[str] = []  # deterministic ordering of bars
    overall: tqdm | None = None
    totals: Dict[str, int] = {}
    dones: Dict[str, int] = {}

    last_refresh = 0.0
    REFRESH_EVERY = 0.05

    while not stop_evt.is_set():
        try:
            msg = q.get(timeout=0.1)
        except pyqueue.Empty:
            msg = None

        if msg is None:
            now = time.time()
            if overall and (now - last_refresh) >= REFRESH_EVERY:
                for b in bars.values():
                    b.refresh()
                overall.refresh()
                last_refresh = now
            continue

        t = msg.get('type')
        if t == 'init':
            sym = msg['symbol']
            total = int(msg['total'])
            totals[sym] = total
            dones.setdefault(sym, 0)
            if sym not in bars:
                position = len(order) + 1  # keep row 0 for the ALL bar
                bars[sym] = tqdm(total=total, desc=sym, ncols=100, position=position, leave=True)
                order.append(sym)
            else:
                bars[sym].total = total

            if overall is None:
                overall = tqdm(total=sum(totals.values()), desc="ALL", ncols=100, position=0, leave=True)
            else:
                overall.total = sum(totals.values())

        elif t == 'tick':
            sym = msg['symbol']
            done = int(msg['done'])
            if sym in bars:
                bars[sym].n = done
            dones[sym] = done
            if overall:
                overall.n = sum(dones.values())

        elif t == 'done':
            sym = msg['symbol']
            if sym in bars:
                bars[sym].n = bars[sym].total
                bars[sym].refresh()

        elif t == 'log':
            tqdm.write(str(msg.get('msg', '')))

        now = time.time()
        if overall and (now - last_refresh) >= REFRESH_EVERY:
            for b in bars.values():
                b.refresh()
            overall.refresh()
            last_refresh = now

    # graceful close
    try:
        for b in bars.values():
            b.close()
        if overall:
            overall.close()
    except Exception:
        pass


def _worker(args):
    """
    Runs in a child process.
    Sends progress messages to the parent via Queue.
    """
    cfg, symbol, q = args
    from src.engine.backtest import run_for_symbol

    def hook(sym: str, done: int, total: int):
        try:
            q.put({'type': 'init', 'symbol': sym, 'total': int(total)})
            q.put({'type': 'tick', 'symbol': sym, 'done': int(done)})
        except Exception:
            pass

    try:
        q.put({'type': 'log', 'msg': f"Starting {symbol}..."})
    except Exception:
        pass

    res = run_for_symbol(cfg, symbol, progress_hook=hook)

    try:
        q.put({'type': 'done', 'symbol': symbol})
    except Exception:
        pass
    return res


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

    # Windows-safe spawn context + Manager Queue
    mp_ctx = mp.get_context("spawn")
    manager = mp_ctx.Manager()
    q = manager.Queue(maxsize=10000)

    stop_evt = threading.Event()
    listener_thread = threading.Thread(target=_progress_listener, args=(q, stop_evt), daemon=True)
    listener_thread.start()

    summaries = []
    try:
        # NOTE: pass mp_context for Windows Python 3.8+ compatibility
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            futs = [ex.submit(_worker, (cfg, sym, q)) for sym in symbols]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    summaries.append(fut.result())
                except Exception as e:
                    summaries.append({'symbol': 'UNKNOWN', 'error': str(e)})
    finally:
        stop_evt.set()
        listener_thread.join(timeout=1.0)

    for s in summaries:
        print(s)


if __name__ == "__main__":
    mp.freeze_support()
    main()
