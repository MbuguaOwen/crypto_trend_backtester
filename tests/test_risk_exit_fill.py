import math


def test_exit_r_long_with_floor(monkeypatch):
    from risk import _exit_fill_price, _clamp, RiskManager  # noqa: F401

    cfg = {
        "risk": {
            "accounting": {"min_r0_bps": 0.5}
        },
        "backtest": {"exits": {"fill_model": "exchange", "slip_bps": 15, "fees_bps_round_trip": 10}}
    }

    rm = RiskManager(cfg, None, None, None)
    trade = {"direction": "LONG", "stop": 95.0, "entry": 100.0, "r0": 0.1, "stop_mode": "SL"}
    row = {"low": 94.5, "high": 101.0, "close": 95.0}
    rm.check_exit(trade, row)

    assert math.isfinite(trade["exit_r"])
    assert 94.5 <= trade["exit"] <= 95.0
    assert trade["exit_r"] < -0.8


def test_exit_r_short_with_floor(monkeypatch):
    from risk import RiskManager

    cfg = {
        "risk": {"accounting": {"min_r0_bps": 0.5}},
        "backtest": {"exits": {"fill_model": "exchange", "slip_bps": 15, "fees_bps_round_trip": 10}}
    }
    rm = RiskManager(cfg, None, None, None)
    trade = {"direction": "SHORT", "stop": 105.0, "entry": 100.0, "r0": 0.1, "stop_mode": "SL"}
    row = {"low": 99.0, "high": 106.0, "close": 105.0}
    rm.check_exit(trade, row)

    assert math.isfinite(trade["exit_r"])
    assert 105.0 <= trade["exit"] <= 106.0
    assert trade["exit_r"] < -0.8

