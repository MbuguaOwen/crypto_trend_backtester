# run_backtest.py
import os
import argparse
import concurrent.futures
import multiprocessing as mp
import queue as pyqueue
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List
import yaml
from tqdm import tqdm
import json

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
    ap.add_argument("--oos_last_k_months", type=int, default=None, help="Hold-out last K months as OOS")
    ap.add_argument("--walkforward", type=str, default=None, help="Walk-forward params 'train=3,test=1,step=1'")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    symbols = list(cfg.get('symbols', []))
    if not symbols:
        print("No symbols specified in config")
        return

    bt_cfg = cfg.get('backtest', {}) or {}
    mode = 'insample'
    oos_k = None
    wf_params = None

    if args.oos_last_k_months is not None and args.walkforward:
        print("Cannot specify both OOS and walk-forward modes")
        return
    if args.oos_last_k_months is not None:
        mode = 'oos'
        oos_k = int(args.oos_last_k_months)
    elif args.walkforward:
        mode = 'walkforward'
        wf_params = {}
        for part in args.walkforward.split(','):
            if '=' in part:
                k, v = part.split('=')
                wf_params[k.strip()] = int(v)
    else:
        mode = bt_cfg.get('mode', 'insample')
        if mode == 'oos':
            oos_k = int(bt_cfg.get('oos_last_k_months', 0))
        elif mode == 'walkforward':
            wf = bt_cfg.get('walkforward', {}) or {}
            wf_params = {
                'train': int(wf.get('train_months', 0)),
                'test': int(wf.get('test_months', 0)),
                'step': int(wf.get('step_months', 0)),
            }

    if mode == 'insample':
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
    elif mode == 'oos':
        from src.engine.data import load_symbol_1m
        from src.engine.walkforward import df_for_months
        from src.engine.backtest import run_for_symbol

        base_out = os.path.join(cfg['paths']['outputs_dir'], 'oos')
        os.makedirs(base_out, exist_ok=True)

        mp_ctx = mp.get_context("spawn")
        manager = mp_ctx.Manager()
        q = manager.Queue(maxsize=10000)
        stop_evt = threading.Event()
        listener_thread = threading.Thread(target=_progress_listener, args=(q, stop_evt), daemon=True)
        listener_thread.start()

        def hook(sym: str, done: int, total: int):
            try:
                q.put({'type': 'init', 'symbol': sym, 'total': int(total)})
                q.put({'type': 'tick', 'symbol': sym, 'done': int(done)})
            except Exception:
                pass

        summaries = []
        agg_keys = [
            'sum_r_sl', 'sum_r_be', 'sum_r_tsl', 'sum_r_sl_overshoot', 'sum_r_realized',
            'SL_count', 'BE_count', 'TSL_count', 'trades'
        ]
        aggregate = {k: 0.0 for k in agg_keys}
        aggregate['SL_count'] = 0
        aggregate['BE_count'] = 0
        aggregate['TSL_count'] = 0
        aggregate['trades'] = 0
        win_acc = 0.0
        avg_acc = 0.0

        try:
            for sym in symbols:
                df_all = load_symbol_1m(cfg['paths']['inputs_dir'], sym, cfg['months'],
                                        progress=cfg['logging']['progress'])
                train_months = cfg['months'][:-oos_k] if oos_k else cfg['months']
                test_months = cfg['months'][-oos_k:] if oos_k else []
                df_train = df_for_months(df_all, train_months)
                df_test = df_for_months(df_all, test_months)
                if df_test.empty:
                    continue
                df_fold = pd.concat([df_train, df_test]).sort_index()
                start_ts = df_test.index[0]
                cfg_sym = deepcopy(cfg)
                cfg_sym['paths']['outputs_dir'] = base_out
                s = run_for_symbol(cfg_sym, sym,
                                   progress_hook=hook,
                                   df1m_override=df_fold,
                                   trade_start_ts=start_ts)
                s.update({
                    'train_months': train_months,
                    'test_months': test_months,
                    'trade_start_ts': start_ts.isoformat(),
                    'bars_train': int(len(df_train)),
                    'bars_test': int(len(df_test)),
                })
                summaries.append(s)
                for k in agg_keys:
                    if k in s:
                        aggregate[k] += s.get(k, 0.0)
                win_acc += s.get('win_rate', 0.0) * s.get('trades', 0)
                avg_acc += s.get('avg_R', 0.0) * s.get('trades', 0)
        finally:
            stop_evt.set()
            listener_thread.join(timeout=1.0)

        tot = aggregate.get('trades', 0)
        if tot > 0:
            aggregate['win_rate'] = win_acc / tot
            aggregate['avg_R'] = avg_acc / tot
        else:
            aggregate['win_rate'] = 0.0
            aggregate['avg_R'] = 0.0
        with open(os.path.join(base_out, 'combined_summary.json'), 'w') as f:
            json.dump({'summaries': summaries, 'aggregate': aggregate}, f, indent=2)
        for s in summaries:
            print(s)
    elif mode == 'walkforward':
        from src.engine.walkforward import run_walkforward

        params = wf_params or {'train': 0, 'test': 0, 'step': 0}

        mp_ctx = mp.get_context("spawn")
        manager = mp_ctx.Manager()
        q = manager.Queue(maxsize=10000)
        stop_evt = threading.Event()
        listener_thread = threading.Thread(target=_progress_listener, args=(q, stop_evt), daemon=True)
        listener_thread.start()

        def hook(sym: str, done: int, total: int):
            try:
                q.put({'type': 'init', 'symbol': sym, 'total': int(total)})
                q.put({'type': 'tick', 'symbol': sym, 'done': int(done)})
            except Exception:
                pass

        try:
            for sym in symbols:
                run_walkforward(cfg, sym,
                                params.get('train', 0),
                                params.get('test', 0),
                                params.get('step', 0),
                                progress_hook=hook)
        finally:
            stop_evt.set()
            listener_thread.join(timeout=1.0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
