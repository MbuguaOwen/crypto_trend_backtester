import os, json, argparse, yaml, traceback, multiprocessing as mp
from tqdm import tqdm
from src.engine.backtest import run_for_symbol as _run_symbol


def run_for_symbol(args):
    symbol, cfg = args
    try:
        summary = _run_symbol(cfg, symbol)
        return symbol, summary
    except Exception as e:
        err = {
            "symbol": symbol,
            "trades": 0,
            "sum_R": 0.0,
            "exits": {},
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()
        }
        return symbol, err


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description="WaveGate Momentum Backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    symbols = cfg['symbols']

    with mp.get_context("spawn").Pool(processes=min(len(symbols), os.cpu_count())) as pool:
        results = list(tqdm(pool.imap_unordered(run_for_symbol, [(s, cfg) for s in symbols]),
                            total=len(symbols), desc="Symbols"))
    combined = {sym: summ for sym, summ in results}

    os.makedirs(cfg['paths']['outputs_dir'], exist_ok=True)
    with open(os.path.join(cfg['paths']['outputs_dir'], "combined_summary.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # Optional: print quick error hints if any symbol had issues
    for sym, summ in combined.items():
        if "error" in summ and summ["error"]:
            print(f"[{sym}] ERROR: {summ['error']} (bars_loaded={summ.get('bars_loaded')})")
