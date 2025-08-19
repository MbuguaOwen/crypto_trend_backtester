import argparse, warnings
from collections import defaultdict
from tqdm import tqdm
from src.engine.backtest import run_all


def main():
    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--suppress-warnings", action="store_true", help="Hide warnings for clean bars")
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore", category=FutureWarning)

    bars = {}
    overall = None
    per_done = defaultdict(int)

    def hook(symbol, done, total):
        if args.no_progress:
            return
        key = symbol.upper()
        if key not in bars:
            bars[key] = tqdm(total=total, desc=f"{key} bars", position=len(bars), ncols=100,
                             mininterval=0.2, smoothing=0.1, leave=True, dynamic_ncols=True)
        # Per-symbol update
        done = max(0, min(done, bars[key].total))
        delta = done - bars[key].n
        if delta > 0:
            bars[key].update(delta)
            per_done[key] = bars[key].n
        # Overall bar
        show_overall = True
        try:
            # prefer YAML flag if available (safe default true)
            from yaml import safe_load
        except Exception:
            pass
        if overall is None:
            total_all = sum(b.total for b in bars.values())
            if total_all > 0:
                overall_desc = "ALL bars"
                overall_position = len(bars) + 1
                overall = tqdm(total=total_all, desc=overall_desc, position=overall_position,
                               mininterval=0.2, smoothing=0.1, leave=True, dynamic_ncols=True)
        if overall is not None:
            # recompute aggregate delta robustly
            agg_done = sum(b.n for b in bars.values())
            delta_overall = agg_done - overall.n
            if delta_overall > 0:
                overall.update(delta_overall)

    run_all(args.config, progress_hook=None if args.no_progress else hook)

    # Close bars
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


if __name__ == "__main__":
    main()
