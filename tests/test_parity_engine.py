import pandas as pd
from backtests.parity_backtest import ParityBacktester, ensure_min_qty_like_ccxt
from regime import TSMOMRegime
from trigger import BreakoutAfterCompression
from risk import RiskManager
from trade import EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL


def test_qty_rounding():
    q = ensure_min_qty_like_ccxt(0.0012, step=0.001, min_amount=0.002, min_cost=5, last_price=10000)
    assert q >= 0.002 and abs((q / 0.001) - round(q / 0.001)) < 1e-9


def test_has_exit_types_in_output(tmp_path, sample_cfg, sample_bars_1m_btc):
    rules = {"BTCUSDT": {"amount_step": 0.001, "min_amount": 0.001, "min_cost": 5}}
    bt = ParityBacktester(sample_cfg, equity_usd=10_000, rules=rules)
    bt.run_symbol("BTCUSDT", sample_bars_1m_btc)
    assert bt.trades, "no trades produced"
    for t in bt.trades:
        assert t["exit_type"] in (EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL)
