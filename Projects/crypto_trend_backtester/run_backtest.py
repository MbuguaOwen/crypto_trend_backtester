from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from engine.data import load_symbol_months
from engine.backtest import run_backtest_symbol
from engine.walkforward import run_walkforward


def parse_walkforward_arg(arg: str) -> Dict[str, int]:
    """
    Parse a string like "train=3,test=1,step=1" into dict.
    """
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    out = {}
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Malformed walkforward spec '{arg}'. Expected 'train=3,test=1,step=1'.")
        k, v = p.split("=", 1)
        out[k] = int(v)
    for k in ("train", "test", "step"):
        if k not in out:
            raise ValueError(f"walkforward spec missing '{k}' in '{arg}'.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Minimal statistical trend-following backtester")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--walkforward", default=None, help='Walk-forward spec, e.g., "train=3,test=1,step=1"')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Deterministic seed (even if unused)
    np.random.seed(int(cfg.get("seed", 42)))

    inputs_dir = cfg["paths"]["inputs_dir"]
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    months: List[str] = list(cfg["data"]["months"])
    symbols: List[str] = list(cfg["symbols"])

    # Walk-forward path
    if args.walkforward:
        wf = parse_walkforward_arg(args.walkforward)
        cfg.setdefault("walkforward", {})
        cfg["walkforward"]["train_months"] = wf["train"]
        cfg["walkforward"]["test_months"] = wf["test"]
        cfg["walkforward"]["step_months"] = wf["step"]

        all_results = {}
        for sym in symbols:
            outdir = outputs_dir / sym / "walkforward"
            stats_list = run_walkforward(load_symbol_months, cfg, sym, outdir=outdir)
            all_results[sym] = stats_list

        print(json.dumps(all_results, indent=2))
        return

    # In-sample path
    results = {}
    for sym in symbols:
        df = load_symbol_months(inputs_dir, sym, months)
        trades, stats = run_backtest_symbol(df, cfg, sym, progress=True)
        results[sym] = stats
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
