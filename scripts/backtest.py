from __future__ import annotations

import os, io, csv, json, argparse, yaml
import numpy as np, pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

from backtester.engine import BacktestEngine
from backtester.metrics import (
    compute_kpis, run_bootstrap_envelopes,
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, white_reality_check
)
from backtester.utils import seed_everything, ensure_dir, save_merged_config, write_advisory

def parse_args():
    ap = argparse.ArgumentParser(description="Crypto Trend Backtester (Streaming, Tick-Driven)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--skip-grid", action="store_true")
    # Optional time slicing overrides (ISO8601, e.g. 2025-07-03T00:00:00Z)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    return ap.parse_args()

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----- Helpers to estimate bar totals (per-file first/last ts) -----
_TS_ALIASES = ("timestamp","time","t","T")

def _robust_parse_ts(s: str) -> pd.Timestamp:
    try:
        v = float(s); unit = "s" if v < 1e11 else "ms"
        return pd.to_datetime(v, unit=unit, utc=True)
    except Exception:
        pass
    out = pd.to_datetime([s], utc=True, format="ISO8601", errors="coerce")[0]
    if pd.isna(out):
        out = pd.to_datetime([s], utc=True, errors="coerce")[0]
    if pd.isna(out):
        raise ValueError(f"Unparseable timestamp: {s}")
    return out

def _detect_ts_idx(header: list[str]) -> int:
    lower = [h.strip().lower() for h in header]
    for cand in _TS_ALIASES:
        for i, h in enumerate(lower):
            if h == cand.lower():
                return i
    raise ValueError(f"Timestamp column not found. Tried {_TS_ALIASES}. Header={header}")

def _read_first_last_ts(path: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = _detect_ts_idx(header)
        first = None
        for row in r:
            if row and any(cell.strip() for cell in row):
                first = row
                break
        if first is None:
            raise ValueError(f"No data rows in {path}")
        ts_first = _robust_parse_ts(first[idx])
    with open(path, "rb") as fb:
        fb.seek(0, os.SEEK_END)
        pos = fb.tell(); buf = b""; last_line = None
        while pos > 0 and last_line is None:
            take = min(4096, pos); pos -= take; fb.seek(pos)
            buf = fb.read(take) + buf
            for line in reversed(buf.split(b"\n")):
                s = line.strip()
                if s:
                    last_line = s.decode("utf-8", "ignore"); break
    sio = io.StringIO(",".join(header) + "\n" + last_line)
    r2 = csv.reader(sio); _ = next(r2); vals = next(r2)
    ts_last = _robust_parse_ts(vals[idx])
    return ts_first, ts_last

def estimate_total_bars(universe_specs, data_dir: str, bar_minutes: int, start: pd.Timestamp|None, end: pd.Timestamp|None) -> int|None:
    total = 0
    for spec in universe_specs:
        path = os.path.join(data_dir, spec["filename"])
        if not os.path.exists(path): continue
        try:
            ts0, ts1 = _read_first_last_ts(path)
            if start is not None and ts0 < start: ts0 = start
            if end is not None and ts1 > end: ts1 = end
            if ts1 <= ts0: continue
            span_min = int(np.ceil((ts1 - ts0).total_seconds() / 60.0))
            total += int(np.ceil(span_min / max(1, bar_minutes)))
        except Exception:
            pass
    return total if total > 0 else None
# -------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.results_dir, run_id)
    ensure_dir(outdir)
    save_merged_config(cfg, os.path.join(outdir, "merged_config.json"))

    universe_specs = [{"symbol": s["symbol"], "filename": s["filename"]} for s in cfg["universe"]]
    streaming_cfg = cfg.get("streaming", {})
    start_iso = args.start or streaming_cfg.get("start")
    end_iso   = args.end   or streaming_cfg.get("end")
    start_ts = pd.to_datetime(start_iso, utc=True) if start_iso else None
    end_ts   = pd.to_datetime(end_iso,   utc=True) if end_iso   else None

    streaming_ctx = {
        "data_dir": args.data_dir,
        "universe": universe_specs,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "checkpoint_stride_bars": int(streaming_cfg.get("checkpoint_stride_bars", 500)),
        "entry_on_bar_close": bool(cfg.get("fast", {}).get("entry_on_bar_close", False)),
    }

    est_total_bars = estimate_total_bars(universe_specs, args.data_dir, cfg.get("bar_interval_minutes",1), start_ts, end_ts)

    param_grid = cfg.get("validation",{}).get("param_grid",{})
    grid_keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]
    grid_rows, grid_returns = [], []

    if grid_keys and not args.skip_grid:
        from itertools import product
        combos = list(product(*grid_values))
        print(f"Running grid scan with {len(combos)} combinations...")
        for i, combo in enumerate(combos, start=1):
            overrides = dict(zip(grid_keys, combo))
            desc = f"Combo {i}/{len(combos)} {overrides}"
            engine = BacktestEngine(cfg, overrides=overrides)
            with tqdm(total=None, desc=desc + " [ticks]", leave=False) as tick_pbar:
                with tqdm(total=est_total_bars, desc=desc + " [bars]", leave=False) as bar_pbar:
                    portfolio = engine.run([], outdir=None, streaming_ctx=streaming_ctx,
                                           tick_pbar=tick_pbar, bar_pbar=bar_pbar)
            rets = portfolio["equity"].set_index("timestamp")["portfolio_equity"].pct_change().fillna(0.0)
            kpis = compute_kpis(rets)
            grid_rows.append({"params": overrides, **kpis})
            grid_returns.append(rets.to_numpy())
        pd.DataFrame(grid_rows).to_csv(os.path.join(outdir, "grid_results.csv"), index=False)
        grid_df = True
    else:
        grid_df = None

    print("Running final backtest...")
    engine = BacktestEngine(cfg, overrides=None)
    with tqdm(total=None, desc="Final [ticks]") as tick_pbar:
        with tqdm(total=est_total_bars, desc="Final [bars]") as bar_pbar:
            portfolio = engine.run([], outdir, streaming_ctx=streaming_ctx,
                                   tick_pbar=tick_pbar, bar_pbar=bar_pbar)

    rets = portfolio["equity"].set_index("timestamp")["portfolio_equity"].pct_change().fillna(0.0)
    kpis = compute_kpis(rets)
    psr = probabilistic_sharpe_ratio(kpis["sharpe"], kpis["n"], sr_bench=0.0, skew=kpis["skew"], kurt=kpis["kurtosis"])
    dsr = deflated_sharpe_ratio(kpis["sharpe"], n_trials=(len(grid_rows) if grid_keys else 1),
                                skew=kpis["skew"], kurt=kpis["kurtosis"], n=kpis["n"])
    kpis.update({"psr": psr, "dsr": dsr})

    boot_cfg = cfg.get("validation",{}).get("bootstrap",{})
    env = run_bootstrap_envelopes(rets.to_numpy(), reps=boot_cfg.get("reps", 1000),
                                  method=boot_cfg.get("method","stationary"),
                                  block_p=boot_cfg.get("block_p",0.1), seed=cfg.get("seed",42)) \
          if boot_cfg.get("enabled", True) else None

    wrc_cfg = cfg.get("validation",{}).get("white_reality_check",{})
    wrc = white_reality_check(grid_returns, reps=1000, block_p=0.1, seed=cfg.get("seed",42)) \
          if (wrc_cfg.get("enabled", True) and grid_keys and len(grid_returns) >= 2 and not args.skip_grid) \
          else {"pvalue": None, "note": "WRC skipped (need >=2 strategies via param_grid)."}

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"kpis": kpis, "wrc": wrc}, f, indent=2)

    if grid_df and not args.no_charts:
        from backtester.sensitivity import tornado_from_grid
        tornado_from_grid(pd.read_csv(os.path.join(outdir, "grid_results.csv")), outdir, metric="calmar")

    if portfolio["trades"].empty:
        write_advisory(os.path.join(outdir, "advisory.txt"), "No trades generated. Loosen thresholds or verify data.")

    if not args.no_charts:
        from backtester.plots import plot_equity_and_drawdown, plot_bootstrap_envelopes
        plot_equity_and_drawdown(portfolio["equity"], outdir)
        if env is not None:
            plot_bootstrap_envelopes(env, outdir)

    print(f"✅ Backtest complete. Artifacts → {outdir}")

if __name__ == "__main__":
    main()
