import pytest
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from engine.risk import (
    RiskCfg,
    update_stops,
    check_exit,
    EXIT_SL,
    EXIT_BE,
    EXIT_TSL,
)


def make_cfg():
    return RiskCfg(
        atr_window=14,
        sl_mode="structure_or_atr",
        sl_atr_mult=1.0,
        be_trigger_r=1.0,
        tsl_start_r=2.0,
        tsl_atr_mult=1.0,
    )


def make_trade(direction="LONG"):
    return {
        "direction": direction,
        "entry": 100.0,
        "stop": 95.0 if direction == "LONG" else 105.0,
        "r0": 5.0,
        "stop_mode": "INIT",
        "be_armed": False,
        "tsl_active": False,
        "be_price": 100.0,
    }


def test_stop_loss_classification():
    trade = make_trade("LONG")
    check_exit(trade, high=101.0, low=94.0)
    assert trade["exit_reason"] == EXIT_SL
    r = (trade["exit"] - trade["entry"]) / trade["r0"]
    assert pytest.approx(r) == -1.0


def test_break_even_classification():
    trade = make_trade("LONG")
    cfg = make_cfg()
    update_stops(trade, price=105.0, atr_last=1.0, cfg=cfg)
    check_exit(trade, high=106.0, low=99.0)
    assert trade["exit_reason"] == EXIT_BE
    r = (trade["exit"] - trade["entry"]) / trade["r0"]
    assert pytest.approx(r) == 0.0


def test_trailing_stop_classification():
    trade = make_trade("LONG")
    cfg = make_cfg()
    update_stops(trade, price=110.0, atr_last=1.0, cfg=cfg)
    check_exit(trade, high=110.0, low=108.0)
    assert trade["exit_reason"] == EXIT_TSL
    r = (trade["exit"] - trade["entry"]) / trade["r0"]
    assert pytest.approx(r) == (109.0 - 100.0) / 5.0
