from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def vol_forecast(returns: pd.Series, cfg: Dict) -> pd.Series:
    method = cfg.get("forecast", {}).get("method", "ewma")
    if method == "ewma":
        alpha = cfg.get("forecast", {}).get("ewma_alpha", 0.06)
        vol = returns.ewm(alpha=alpha, adjust=False).std().shift(1)
    else:
        window = cfg.get("forecast", {}).get("window", 60)
        vol = returns.rolling(window).std().shift(1)
    floor = cfg.get("forecast", {}).get("floor_sigma", 0.0005)
    return vol.clip(lower=floor)


def position_notional(price: float, sigma: float, cfg: Dict) -> float:
    target = cfg.get("target_sigma_annual", 0.12)
    base = cfg.get("per_symbol_notional_cap_usd", 20000)
    ann_factor = np.sqrt(365 * 24 * 60)
    forecast_ann = sigma * ann_factor
    notional = target / forecast_ann * base if forecast_ann > 0 else base
    cap = cfg.get("per_symbol_notional_cap_usd", 20000)
    return float(np.clip(notional, -cap, cap))
