#!/usr/bin/env python3
"""
Compare entry modes (BOCPD-only build).
Currently supports only "bocpd_squeeze_breakout"; legacy modes are skipped.
"""
import os, copy, argparse, pandas as pd
from .parity_backtest import _load_yaml, run_symbol, summarize

def run_mode(cfg_path, symbol, mode):
    cfg = _load_yaml(cfg_path)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("entry", {})["mode"] = mode

    out_dir = os.path.join("outputs", mode)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}_trades_parity.csv")

    df = run_symbol(cfg, symbol, out_path)
    summary = summarize(df, symbol, out_dir=out_dir)
    summary["mode"] = mode
    return out_path, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--modes", nargs="+", default=["bocpd_squeeze_breakout"])
    args = ap.parse_args()

    rows=[]
    for m in args.modes:
        if m != "bocpd_squeeze_breakout":
            print(f"Skipping legacy mode '{m}' (BOCPD-only build).")
            continue
        print(f"\n=== Running mode: {m} ===")
        _, summ = run_mode(args.config, args.symbol, m)
        rows.append(summ)

    rep = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    rep.to_csv(os.path.join("outputs", f"{args.symbol}_entry_compare.csv"), index=False)
    print("\nSaved comparison →", os.path.abspath(os.path.join("outputs", f"{args.symbol}_entry_compare.csv")))
    print(rep.to_string(index=False))

if __name__ == "__main__":
    main()

