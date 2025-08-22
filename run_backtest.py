import argparse
import warnings
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.engine.backtest import run_for_symbol, load_config


def _run_symbol(args):
    cfg, symbol = args
    return run_for_symbol(cfg, symbol, progress_hook=None)


def main() -> None:
    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument(
        "--parallel",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use process pool across symbols (auto: on when no-progress)",
    )
    ap.add_argument(
        "--suppress-warnings", action="store_true", help="Hide warnings for clean bars"
    )
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)

    cfg = load_config(args.config)
    symbols = cfg["symbols"]

    use_progress = not args.no_progress
    want_parallel = args.parallel in ("on", "auto") and not use_progress

    if want_parallel:
        summaries = []
        pbar = tqdm(total=len(symbols), desc="Symbols")
        with ProcessPoolExecutor(
            max_workers=min(os.cpu_count() or 2, len(symbols))
        ) as ex:
            futures = {ex.submit(_run_symbol, (cfg, sym)): sym for sym in symbols}
            for fut in as_completed(futures):
                summaries.append(fut.result())
                pbar.update(1)
        pbar.close()
    else:
        summaries = []
        overall = tqdm(total=len(symbols), desc="Symbols", disable=not use_progress)
        for sym in symbols:
            bar = None

            def hook(symbol: str, done: int, total: int) -> None:
                nonlocal bar
                if bar is None:
                    bar = tqdm(total=total, desc=f"{symbol} bars", leave=True)
                delta = done - bar.n
                if delta > 0:
                    bar.update(delta)

            res = run_for_symbol(cfg, sym, progress_hook=hook if use_progress else None)
            summaries.append(res)
            if bar is not None:
                bar.close()
            overall.update(1)
        overall.close()

    outdir = cfg["paths"]["outputs_dir"]
    os.makedirs(outdir, exist_ok=True)
    print("Done. Summaries:", summaries)


if __name__ == "__main__":  # pragma: no cover
    main()

