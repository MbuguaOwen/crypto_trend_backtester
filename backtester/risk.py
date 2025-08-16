# backtester/risk.py
from __future__ import annotations

def vol_target_notional(equity: float, target_annual_vol: float, vol_forecast: float, cap: float | None = None) -> float:
    if vol_forecast <= 0:
        return 0.0
    notional = (target_annual_vol / vol_forecast) * equity
    if cap is not None:
        notional = min(notional, cap * equity)
    return max(0.0, notional)
