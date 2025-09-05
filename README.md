# Minimal Statistical Trend-Following Engine

A compact, deterministic, walk-forward-capable trend-following backtester built on **pure statistics**:
- **Macro regime** via t‑stat of mean 1‑minute log returns (κ).
- **Micro alignment + acceleration** via short/long t‑stats (κ_short − κ_long).
- **Ignition** on **Donchian breakout** with an **ATR buffer** so we don’t buy pullbacks.
- **Risk**: fixed ATR stop; optional **breakeven**; **no trailing** (discovery mode).
- **Walk‑forward** (train *k* months → test 1 month → step 1) with **no leakage**.
- **Outputs** per fold: `stats.json` and `trades.csv`.

## Data layout
Place your 1‑minute OHLCV CSVs here (one file per month):
```
inputs/<SYMBOL>/<YYYY-MM>.csv
```
**Required columns** (lowercase): `timestamp,open,high,low,close,volume`  
`timestamp` can be **ms since epoch** or **ISO**. Data is treated as **UTC**.

Rules:
- Index is strictly increasing and de‑duplicated (keep last).
- No gap filling or inference—data is used **as is**.

## Quickstart
1) Install dependencies (Python 3.10+):
```
pip install -r requirements.txt
```

2) Adjust `configs/default.yaml` thresholds, months, and symbols to your needs.

### In‑sample (dev) run
```
python run_backtest.py --config configs/default.yaml
```
Prints JSON stats for each symbol and uses **all months** in the config.

### Walk‑forward (WF) run
```
python run_backtest.py --config configs/default.yaml --walkforward "train=3,test=1,step=1"
```
Writes per‑fold outputs to:
```
outputs/<SYMBOL>/walkforward/fold_<N>/{stats.json,trades.csv}
```
and prints a JSON summary to the console.

## What to expect
- **Stats JSON keys**: `symbol, trades, win_rate, avg_R, sum_R, blockers`.
- **Trades CSV columns**: `side,entry_ts,exit_ts,entry,exit,R` (R is P&L divided by ATR‑based risk at entry).
- Deterministic behavior (global seed set) and **progress bars** via `tqdm`.

## Allowed libraries
Exactly: `numpy`, `pandas`, `pyyaml`, `tqdm` (plus Python stdlib).

---

**Note:** This is a minimal discovery engine—by design there is **no trailing stop** and trades exit at **SL** or **end‑of‑run**. Breakeven is optional.
