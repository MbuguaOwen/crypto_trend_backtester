
# WaveGate Momentum Backtester

**Stack**
- Macro Regime: Multi-horizon TSMOM (4–5 lookbacks)
- Waves: Event-time CUSUM bars (volatility-normalized)
- Trigger: 1m power bar (z-score + body dominance + TR/ATR expansion)
- Risk: ATR(1m) initial stop; adaptive BE + TSL thresholds

**Design Goals**
- Single source of truth for adaptivity (`AdaptiveController`) – no silent fallbacks; warm-start before first trade.
- Minimal, non-duplicated config surface with only high-signal knobs exposed.
- Ticks → 1m OHLCV ingest; tz-aware UTC; robust resampling.

## Quick Start
1. Drop monthly tick CSVs under `inputs/<SYMBOL>/<SYMBOL>-ticks-YYYY-MM.csv`
   with columns: `timestamp,price,qty,is_buyer_maker` (timestamp in **ms**).
2. Edit `configs/default.yaml` (symbols, months, paths).
3. Run:
   ```bash
   python -m pip install -r requirements.txt
   python run_backtest.py --config configs/default.yaml --workers 1
   ```
   Use `--workers 0` to use all cores (parallel symbols).

**Per-bar progress** prints when `--workers 1`. When parallel, you’ll see overall task progress and per-month ingest bars.

## Outputs
- `outputs/<SYMBOL>_trades.csv`
- `outputs/<SYMBOL>_summary.json`
- `outputs/logs/<SYMBOL>_params_run.json`

## Fixed vs Adaptive
- Default is **adaptive** with strict ranges.
- To run fixed baselines, set `adaptive.enabled: false` in the relevant blocks and set `fixed` values.
