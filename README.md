# Multi-Horizon Crypto Backtesting System (TSMOM)

A production-grade, event-driven backtester for multi-horizon trend-following with volatility targeting and breakout confirmation.
Designed to run on 7 months of **tick data (Jan–Jul 2025)** for **ETHUSDT, BTCUSDT, SOLUSDT** and **never** crash with `KeyError: 'qty'`.

---

## ✅ Key Guarantees

- **No `KeyError: 'qty'`**: The new `data_loader.py` maps *any* reasonable quantity/volume column to `qty`, e.g. `quantity`, `volume`, `size`, `vol`, `q`, and can **derive** it from `quote_qty / price` when only notional is present.
- **Schema-flexible loader**: Robust column detection for `timestamp`, `price`, `qty`, and `is_buyer_maker` with thorough validation & cleaning.
- **Event-driven engine**: Simulated BAR ➜ SIGNAL ➜ ORDER ➜ FILL flow with **fees, slippage, and latency**.
- **Warmup gate**: No signal generation or trading until **all lookback windows** (momentum/vol/breakout) are fully populated.
- **Progress bars**: TQDM bars for loading and for backtest processing per symbol.

---

## 📂 Project Structure

```
backtester/
  __init__.py
  engine.py
  portfolio.py
  execution.py
  events.py
  utils/
    __init__.py
    math.py
    time.py
backtests/
  parity_backtest.py
data_loader.py
strategy.py
configs/
  backtest.yaml
data/
  BTCUSDT/ ETHUSDT/ SOLUSDT/
results/
README.md
```

---

## 📊 Expected Tick CSV Schema

The loader is **flexible**. It tries to find columns **case-insensitively**:

- **timestamp**: one of `timestamp`, `ts`, `time`, `T`, `event_time`, `trade_time_ms`, `trade_time`
- **price**: one of `price`, `p`, `last_price`, `rate`
- **qty (base units)**: one of `qty`, `quantity`, `amount`, `size`, `vol`, `volume`, `q`, `trade_quantity`, `last_qty`
- **quote_qty (notional)**: one of `quote_qty`, `quoteQuantity`, `qv`, `amount_quote`, `notional`
- **maker flag (optional)**: one of `is_buyer_maker`, `isBuyerMaker`, `maker`, `is_maker`, `is_seller_maker`

If **`qty` is missing**:
- If we have `quote_qty` **and** `price`, we compute `qty = quote_qty / price`.
- Otherwise, the file is flagged with a clear error message (not a `KeyError`), pointing to the missing fields.

The CSV filename can be anything. We recommend putting files under:
```
data/{SYMBOL}/*2025-01*.csv
data/{SYMBOL}/*2025-02*.csv
...
data/{SYMBOL}/*2025-07*.csv
```
The loader scans all CSVs under each symbol directory and uses an internal filter for months.

---

## 🧮 Strategy (TSMOM + Breakout + Vol Target)

- **Momentum**: sign-weighted composite of multiple lookbacks (minutes). Example: `[60, 240, 720, 1440]`.
- **Breakout confirmation**: Donchian channel over `breakout_lookback` bars; trades only if price breaks the prior high/low consistent with momentum sign.
- **Vol targeting**: Scales position size using realized volatility (EWMA of log-returns) to target annualized volatility. Clipped by `max_leverage`.

> **Safety gate**: No signals until `max(momentum_windows) + breakout_lookback + vol_window` bars have occurred.

---

## ⚙️ Configuration

Edit `configs/backtest.yaml`. Key fields:

```yaml
data:
  data_dir: ./data
  symbols: [BTCUSDT, ETHUSDT, SOLUSDT]
  months: ["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07"]
  resample_interval: "1min"

broker:
  initial_capital: 100000
  taker_fee_bps: 7.5
  slippage_bps: 1.5
  latency_ms: 500

strategy:
  momentum_windows: [60, 240, 720, 1440]
  breakout_lookback: 300
  vol_window: 1440
  target_vol_annual: 0.30
  max_leverage: 3.0
```

---

## ▶️ How to Run

1. **Install deps**:

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pandas numpy pyyaml tqdm
```

2. **Place your tick CSVs** under `data/{SYMBOL}/` for the months Jan–Jul 2025.

3. **Run**:

```bash
python backtest_multi_horizon.py --config configs/backtest.yaml --results-dir results
```

You’ll see progress bars for loading and simulation. Outputs:
- `results/{SYMBOL}_trades.csv`
- `results/{SYMBOL}_equity.csv`
- `results/summary.csv`

---

## 🧰 How `KeyError: 'qty'` Was Solved

**Root cause:** Hard-coded assumption that the volume column is named `qty`.

**Solution:** In `data_loader.py`, we:
1. **Map columns case-insensitively** from a set of known aliases (see above).
2. **Derive** `qty` as `quote_qty / price` if only notional is present.
3. **Validate** and **sanitize** data (drop NaNs, non-positive price/qty, de-duplicate).
4. **Standardize** to canonical columns: `ts` (int ms), `price` (float), `qty` (float), `is_buyer_maker` (bool).
5. Build **minute bars** consistently across all symbols.

Therefore, no `KeyError` occurs; if essential fields are missing, the loader raises a **clear, actionable ValueError** with context.

---


## 🎯 Parity Backtest

A discrete-trade backtester that mirrors the live engine. Example usage:

```bash
python -m backtests.parity_backtest \
  --config config.yaml \
  --symbol BTCUSDT \
  --data_1m_csv data/BTCUSDT_1m.csv \
  --out trades_parity.csv \
  --equity_usd 10000
```

## 🧪 Notes

- The backtester is **event-driven** and **symbol-isolated** for clarity. Extending to cross-asset synchronization is straightforward.
- Signals are **suppressed during warmup** by design.
- If a month has no data, the engine skips gracefully with a warning.

Happy testing. Ship alpha. 🚀
