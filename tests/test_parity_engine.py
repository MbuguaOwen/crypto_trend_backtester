import pandas as pd
import numpy as np
from core_reuse.execution_helpers import ensure_min_qty
from core_reuse.trade import EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL
from backtests.parity_backtest import ParityEngine
import yaml

def test_qty_rounding_meets_minimums():
    amt, ok = ensure_min_qty(0.00019, 60000, amount_step=0.0001, min_qty=0.0001, min_notional=5.0)
    assert ok is True and abs(amt-0.0001)<1e-12

    amt, ok = ensure_min_qty(0.00005, 60000, amount_step=0.0001, min_qty=0.0001, min_notional=5.0)
    assert ok is False

    amt, ok = ensure_min_qty(0.0001, 1.0, amount_step=0.0001, min_qty=0.0001, min_notional=5.0)
    assert ok is False  # notional too small

def test_exit_taxonomy_is_valid():
    assert set([EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL]) == {"SL","TP","BE","TSL"}

def test_synthetic_bars_produce_entry_and_exit(tmp_path):
    cfg = yaml.safe_load(open("configs/config.yaml","r",encoding="utf-8"))
    sym = "BTCUSDT"
    idx = pd.date_range("2025-01-01", periods=400, freq="1min", tz="UTC")
    price = np.concatenate([
        np.full(200, 100.0),
        np.linspace(100, 110, 200)
    ])
    df = pd.DataFrame({
        "open": price,
        "high": price + 0.05,
        "low":  price - 0.05,
        "close": price,
        "volume": 1.0
    }, index=idx)
    out = tmp_path / "trades.csv"
    eng = ParityEngine(cfg, sym, equity_usd=10000, progress=False)
    eng.run(df, str(out))
    rows = pd.read_csv(out)
    assert len(rows) >= 1
    assert rows["exit_type"].isin(["SL","TP","BE","TSL"]).all()
