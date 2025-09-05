# engine/walkforward.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .data import load_symbol_months
from .backtest import run_insample

def month_slices(months: list[str], train: int, test: int, step: int):
    i = 0
    while i + train + test <= len(months):
        yield months[i:i+train], months[i+train:i+train+test]
        i += step

def run_walkforward(symbol: str, cfg: dict, outroot: Path):
    months = cfg["data"]["months"]
    train_n = cfg["walkforward"]["train_months"]
    test_n = cfg["walkforward"]["test_months"]
    step_n = cfg["walkforward"]["step_months"]

    fold = 0
    results = []
    for train_m, test_m in month_slices(months, train_n, test_n, step_n):
        # Train phase is feature/stat estimation only; we don’t fit ML, so we just load to respect no-leakage windows
        df_test = load_symbol_months(cfg["paths"]["inputs_dir"], symbol, test_m)

        outdir = outroot / "walkforward" / f"fold_{fold:02d}"
        stats = run_insample(df_test, symbol, cfg, outdir)
        stats["fold"] = fold
        stats["train_months"] = train_m
        stats["test_months"] = test_m
        results.append(stats)
        fold += 1

    pd.DataFrame(results).to_csv(outroot / "walkforward" / "fold_stats.csv", index=False)
    return results
