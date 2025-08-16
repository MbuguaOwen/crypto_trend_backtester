# TSMOM Parity Backtest

A discrete-trade, 1:1 parity backtester that mirrors the live TSMOM engine’s logic:
regime (multi-horizon), compression-gated Donchian / k-sigma trigger, ATR risk, and
SL/TP/BE/TSL exits — including exchange minQty/step/minNotional enforcement.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m backtests.parity_backtest   --config configs/config.yaml   --symbol BTCUSDT   --data data/BTCUSDT/sample_1m.csv   --out outputs/BTCUSDT_trades.csv   --equity_usd 10000
```

To test tick ingestion (aggregation to 1m):
```bash
python -m backtests.parity_backtest   --config configs/config.yaml   --symbol BTCUSDT   --data data/BTCUSDT/sample_ticks.csv   --out outputs/BTCUSDT_trades_from_ticks.csv   --equity_usd 10000
```

Run tests:
```bash
pytest -q
```

## Notes

- Uses **prior-bar** Donchian levels plus ATR buffer. No same-bar peeking.
- Trailing stop and breakeven rules mirror live semantics.
- Quantity is rounded and validated against step, minQty, and minNotional.
- Deterministic: given same inputs and config, output is stable.
- Progress bars (tqdm) show file/bars ETA; turn off with `--no-progress`.


## Using Your Live Modules Directly (1:1 Parity)

If your live engine exposes these exact classes:

- `TSMOMRegime`
- `BreakoutAfterCompression`
- `RiskManager`
- `Trade`
- `CcxtExchange._ensure_min_qty` (or equivalent)

You can wire them in *without* changing the backtester:

1) Add your live project root to `PYTHONPATH`, e.g.
```bash
export PYTHONPATH=/path/to/tsmom_live_system:$PYTHONPATH
```

2) In `core_reuse/` files, replace the reference implementations with direct imports, for example:
```python
# core_reuse/regime.py
from your_live_package.core.regime import TSMOMRegime  # noqa: F401
```

Do the same for `trigger.py`, `risk.py`, `trade.py`, and `execution_helpers.py` (import your
min-qty rounding helper). The rest of the backtester will use them unchanged.
