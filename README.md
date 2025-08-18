# TSL-Accurate Backtest (Walk-Forward Safe)

This mini-project implements a tick→bar backtest with **profit-protective TSL** that obeys the invariants:

- **Invariant A (profit protective)**: For TSL exits, trail is always on the profit side of entry (fee-buffer aware)
- **Invariant B (precedence)**: SL → BE → TSL → TP (deterministic intrabar checks)
- **Invariant C (state)**: TSL can only fire if `trail_active == True` and activation requires reaching `trail_activation_r` first

It logs **true R** metrics when possible and includes:
- `initial_sl`, `tp`, `risk_r_denom`, `mfe_price`, `mfe_r`, `r_at_exit`,
- `trail_activated_at_r`, `trail_at_exit`, and a clamp audit

## Run

```bash
python -m backtests.backtest_tick   --data-dir tick_data   --results-dir results   --pairs BTCUSDT SOLUSDT   --months 2025-07   --config configs/default.yaml   --dump-config
```

### Tick CSV format (per month)
Assumes files like: `tick_data/BTCUSDT-ticks-YYYY-MM.csv` with columns:
- `timestamp` (ms or ISO), `price`, `qty`  (extra columns are ignored)

### Notes
- Orders execute **on next bar open** after a bar-close signal.
- Intrabar exits use **H/L** with precedence SL→BE→TSL→TP for longs (mirror for shorts).
- Fee model uses `fee_bps` (applied on entry and exit).

