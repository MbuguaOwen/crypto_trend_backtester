# Crypto Trend Backtester — WaveGate Momentum (Jan–Jul 2025)

One-file command to run:
```
python run_backtest.py --config configs/default.yaml
```

## What this does
- Loads **tick CSVs** from `inputs/<SYMBOL>/<SYMBOL>-ticks-YYYY-MM.csv`.
- Aggregates to **1m bars**, derives **5m/15m** on the fly.
- **Gate #1:** Multi‑horizon TSMOM vote (1m/5m/15m/1h) → LONG/SHORT/FLAT.
- **Gate #2:** **WaveGate** (5m) detects W1→W2 and arms only when W2 termination zone is likely complete.
- **Trigger:** 1m **Momentum Ignition** → break of W2 high/low + thrust (z‑return, body dominance, TR/ATR).
- **Risk:** SL at W2 floor (or ATR), BE @ +0.5R, TSL from +1.0R.
- Writes results to `outputs/`, e.g. `<SYMBOL>_trades.csv` and a JSON summary.

> Built to be **causal** (no repaint), simple, and fast to iterate.

## Tick CSV format
Expected columns (case-insensitive, any subset ok):
- `timestamp` (ms or ISO), `price`, `qty` **or** (`amount`, `size`, `volume`).
- Extra columns are ignored.

Example header:
```
timestamp,price,qty,is_buyer_maker
```

## Requirements
```
python>=3.9
pandas
numpy
pyyaml
tqdm
```
Install:
```
pip install -r requirements.txt
```

## Config
See `configs/default.yaml` for symbols, months, thresholds, and all knobs. Jan–Jul 2025 are prefilled. Adjust as needed.

## Outputs
- `outputs/<SYMBOL>_trades.csv` — per-trade log with entry/exit and exit reason (SL/TP/BE/TSL)
- `outputs/<SYMBOL>_summary.json` — counts, win rate, expectancy, and R-metrics

## Notes
- Warmup bars enforced to avoid partial windows.
- The WaveGate here is intentionally lean. It uses an ATR‑ZigZag and objective depth/timing heuristics to identify W1/W2 without repainting.
- Start with defaults; once you see healthy flow, dial thresholds or layer BOCPD/Donchian refinements.
