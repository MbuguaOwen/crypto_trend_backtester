import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def sample_cfg():
    return {
        "strategy": {
            "tsmom_regime": {"timeframes": {"1m": {}}},
            "trigger": {"breakout": {"donchian_lookback": 3, "buffer_atr_mult": 0.0}},
        },
        "risk": {
            "atr": {"window": 1},
            "stops": {
                "initial_sl_atr_mult_default": 1.0,
                "take_profit_atr_mult": 2.0,
                "move_to_breakeven": {"trigger_r_multiple": 1e9, "be_offset_atr": 0.0},
                "trailing": {"trail_atr_mult_default": 0.0, "step_atr": 0.0},
            },
            "vol_targeting": {
                "sizing": {"per_trade_risk_cap_pct_equity": 0.01, "max_leverage": 5.0},
                "per_symbol_notional_cap_pct_equity": 1.0,
            },
        },
        "exchange": {"taker_fee_bps": 7.5},
        "symbols": {"BTCUSDT": {}},
    }


@pytest.fixture
def sample_bars_1m_btc():
    idx = pd.date_range("2024-01-01", periods=7, freq="1min", tz="UTC")
    data = {
        "open": [100, 100, 100, 100, 100, 100, 102],
        "high": [101, 101, 101, 101, 101, 102, 107],
        "low": [99, 99, 99, 99, 99, 100, 102],
        "close": [100, 100, 100, 100, 100, 102, 107],
        "volume": [1] * 7,
    }
    df = pd.DataFrame(data, index=idx)
    return df
