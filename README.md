# TSMOM Parity Backtest — Live 1:1 Mirror (Harness)

This zip contains a thin, deterministic **backtest harness** that executes your **live modules** on historical 1‑minute bars, preserving **order-of-operations** and exit taxonomy (**SL / TP / BE / TSL**).

> **Parity principle:** Swap the files in `core_reuse/` with your *actual live* `regime.py`, `trigger.py`, `risk.py`, `trade.py`, and the real helper that mirrors `CcxtExchange._ensure_min_qty` to achieve **bit-for-bit parity**. The bundled versions are faithful **shims** so this zip runs locally without secrets.

## Quickstart

```bash
# 1) (optional) create venv
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) run default (Jan–Jul 2025 from configs/default.yaml), all symbols
python -m backtests.parity_backtest --config configs/default.yaml

# 4) run single symbol and custom out
python -m backtests.parity_backtest --config configs/default.yaml --symbol BTCUSDT --out outputs/BTCUSDT_trades_parity.csv
```

If no monthly files exist at `backtest.inputs.path_pattern`, the harness falls back to `inputs/{symbol}/sample_1m.csv` so you can test the flow end-to-end.

## What to swap for **true parity**

Replace the shims in `core_reuse/` with your live engine modules:

- `regime.py` → contains `class TSMOMRegime`
- `trigger.py` → contains `class BreakoutAfterCompression` (Donchian **prior-bar** + ATR buffer; compression gate)
- `risk.py` → `class RiskManager` with your ATR + sizing + initial SL/TP
- `trade.py` → `class Trade` with **BE** and **TSL** lifecycle
- `execution_helpers.py` → factor out live `_ensure_min_qty` math and import here

No other changes required. The harness calls the exact interface you specified.

## Config (single source of truth)

Edit `configs/default.yaml`. It contains **strategy**, **risk**, **execution**, and **backtest** blocks. The backtest window defaults to **2025‑01‑01 → 2025‑08‑01** (Jan–Jul 2025). Toggle the progress bar via `backtest.progress_bar`.

Per‑symbol exchange constraints (minQty, step, minNotional) are provided under `symbols.*` and are consumed by `ensure_min_qty_like_live(...)` in backtests.

## Data

Set:
- `backtest.inputs.type`: `ohlcv_csv` (default) or `tick_csv`
- `backtest.inputs.path_pattern`: e.g. `data/{symbol}/1m/{yyyymm}.csv`
- `backtest.inputs.tick_path_pattern`: e.g. `data/{symbol}/ticks/{yyyymm}.csv`

The loader stitches monthly shards across the configured start/end and **clips** to the window. All timestamps are treated as **UTC**.

## Determinism and **no‑peek** rules

- Work stream = **1m bars**; downsampling builds regime timeframes from the same history **≤ current bar**, never future data.
- **Donchian** uses **prior bar** high/low plus an **ATR buffer** (as in live).
- Entries are discrete; at most **one open trade** per symbol.
- **BE/TSL** maintenance occurs **before** exit checks on every bar.
- Fees applied on both entry and exit. Slippage is deterministic if configured (default 0 bps).

## Output schema

One row per closed trade:

```
ts_open, ts_close, symbol, side, entry, exit, qty, pnl, exit_type, entry_reason
```

`exit_type ∈ {SL, TP, BE, TSL}`, `entry_reason ∈ {"donchian_breakout","ksigma_breakout"}`.

## Tests

Run `pytest` to validate warmup gating and basic taxonomy. Tests pass with the shims; they’ll remain green when you swap in your live modules if interfaces match.

## Notes

- To mirror **next bar** fills, set `backtest.simulator.entry_fill: next_open`.
- For exact parity on exchange constraints, port your live `_ensure_min_qty` math into `core_reuse/execution_helpers.py`.
- If your live stops/trailing use per‑symbol overrides, place them under `symbols.<SYM>.stops` in the config.
