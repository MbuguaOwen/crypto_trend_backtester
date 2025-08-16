# Crypto Trend Backtester — Streaming (Tick-Driven) Edition

Implements the blueprint strategy with **tick ingestion**, internal **bar aggregation**, and **event-driven** logic:
- **Regime:** Multi-horizon TSMOM (from completed bars)
- **Trigger:** Donchian or K-sigma breakout, **gated by compression** (BB-width percentile)
- **Risk:** Vol targeting + ATR SL/TP + breakeven + optional trailing
- **Validation:** KPIs, PSR/DSR, White’s Reality Check (WRC), bootstrap P&L envelopes, tornado plots

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data (TICK CSVs)
Place per-symbol **tick** CSVs in `data/` with columns:
```
timestamp,price,quantity,is_buyer_maker,a
# timestamp can be ISO8601 UTC or ms since epoch
```
The engine aggregates ticks into OHLCV bars internally (default: 1-minute) and evaluates signals on **every tick** using thresholds from the **latest completed bars**.

## Run
```bash
python scripts/backtest.py --config configs/default.yaml --data-dir data --results-dir results
```

Artifacts in `results/<RUN_ID>/`:
- `merged_config.json` — frozen config
- `trades.csv`, `fills.csv` — all orders/events
- `equity.csv` — portfolio equity (recorded at bar_close by default)
- `metrics.json` — KPIs + PSR/DSR + White’s RC p-value
- `grid_results.csv` — param scan results (if enabled)
- `charts/` — equity, drawdown, bootstrap envelopes, tornado
- `advisory.txt` — zero-trade or validation notes

## Streaming mode (25M+ ticks/month ready)
- Reads CSVs with `chunksize` and merges via **k-way time merge** across symbols.
- Indicators compute **on bar completion** only (fast). Entries/exits checked on **every tick**.
- Equity recording defaults to **bar_close** (keeps artifacts tiny). Switch to interval-based in YAML.
