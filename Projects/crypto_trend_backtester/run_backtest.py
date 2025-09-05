# run_backtest.py
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, yaml

from engine.data import load_symbol_months
from engine.backtest import run_insample
from engine.walkforward import run_walkforward

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--walkforward", default="", help='format "train=3,test=1,step=1" to override YAML')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg.get("seed", 42)))

    if args.walkforward:
        parts = dict(kv.split("=") for kv in args.walkforward.split(","))
        cfg["walkforward"]["train_months"] = int(parts.get("train", cfg["walkforward"]["train_months"]))
        cfg["walkforward"]["test_months"]  = int(parts.get("test",  cfg["walkforward"]["test_months"]))
        cfg["walkforward"]["step_months"]  = int(parts.get("step",  cfg["walkforward"]["step_months"]))

    inputs = cfg["paths"]["inputs_dir"]
    outputs = Path(cfg["paths"]["outputs_dir"]); outputs.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for sym in cfg["symbols"]:
        outroot = outputs / sym
        if args.walkforward or cfg.get("walkforward"):
            res = run_walkforward(sym, cfg, outroot)
            all_stats.extend(res)
        else:
            df = load_symbol_months(inputs, sym, cfg["data"]["months"])
            stats = run_insample(df, sym, cfg, outroot)
            all_stats.append(stats)

    (outputs / "summary.json").write_text(json.dumps(all_stats, indent=2))
    print(json.dumps(all_stats, indent=2))

if __name__ == "__main__":
    main()
