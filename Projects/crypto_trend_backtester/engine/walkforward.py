from __future__ import annotations

from typing import Dict, Generator, Iterable, List, Tuple, Callable
import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest_symbol


def split_months(months: List[str], train: int, test: int, step: int) -> Generator[Tuple[int, List[str], List[str]], None, None]:
    """
    Yield (fold_idx, train_months, test_months) moving forward by `step`.
    """
    m = len(months)
    fold = 1
    i = train
    while i + test <= m:
        train_slice = months[i - train: i]
        test_slice = months[i: i + test]
        yield fold, train_slice, test_slice
        fold += 1
        i += step


def run_walkforward(
    loader_fn: Callable[[str, str, List[str]], pd.DataFrame],
    cfg: Dict,
    symbol: str,
    outdir: str | Path,
) -> List[Dict]:
    """
    Run walk-forward folds and write per-fold stats/trades under outdir.
    Returns list of per-fold stats dicts.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    months: List[str] = list(cfg["data"]["months"])
    train_n = int(cfg["walkforward"]["train_months"])
    test_n = int(cfg["walkforward"]["test_months"])
    step_n = int(cfg["walkforward"]["step_months"])

    all_stats: List[Dict] = []

    for fold, train_months, test_months in split_months(months, train_n, test_n, step_n):
        # Load train+test concatenated to avoid leakage in features
        df = loader_fn(cfg["paths"]["inputs_dir"], symbol, train_months + test_months)

        # Determine test window boundaries
        test_start = pd.Timestamp(test_months[0] + "-01")
        # Approximate end: next month first day minus a tiny delta
        # We'll derive end by finding the max timestamp within the union of test months
        test_df = df[df.index.to_period("M").astype(str).isin(test_months)]
        if test_df.empty:
            # skip empty test
            continue
        test_start_ts = test_df.index.min()
        test_end_ts = test_df.index.max()

        trades, stats = run_backtest_symbol(
            df, cfg, symbol, progress=True, test_start=test_start_ts, test_end=test_end_ts
        )
        stats_out = {
            **stats,
            "fold": fold,
            "train_months": train_months,
            "test_months": test_months,
        }
        all_stats.append(stats_out)

        fold_dir = outdir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Write stats.json
        (fold_dir / "stats.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
        # Write trades.csv
        import pandas as pd  # local import; stdlib allowed
        trades_df = pd.DataFrame(trades, columns=["side", "entry_ts", "exit_ts", "entry", "exit", "R"])
        trades_df.to_csv(fold_dir / "trades.csv", index=False)

    return all_stats
