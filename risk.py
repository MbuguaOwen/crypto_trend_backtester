from typing import Dict, Tuple
import pandas as pd


class RiskManager:
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    # --- ATR ---
    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        window = self.cfg.get("risk", {}).get("atr", {}).get("window", 14)
        h = df["high"]
        l = df["low"]
        c = df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        return atr

    # --- initial stop/tp ---
    def initial_levels(self, side: str, price: float, atr: float, symcfg: Dict) -> Tuple[float, float]:
        stops_cfg = self.cfg.get("risk", {}).get("stops", {})
        sl_mult = stops_cfg.get("initial_sl_atr_mult_default", 1.0)
        tp_mult = stops_cfg.get("take_profit_atr_mult", 2.0)
        if side == "long":
            sl = price - atr * sl_mult
            tp = price + atr * tp_mult
        else:
            sl = price + atr * sl_mult
            tp = price - atr * tp_mult
        return sl, tp

    # --- position sizing ---
    def position_size_units(self, equity_usd: float, entry: float, sl: float, fees_bps: float, side: str, max_leverage: float) -> float:
        risk_cfg = self.cfg.get("risk", {}).get("vol_targeting", {}).get("sizing", {})
        risk_cap_pct = risk_cfg.get("per_trade_risk_cap_pct_equity", 0.01)
        risk_per_trade = equity_usd * risk_cap_pct
        risk_per_unit = abs(entry - sl)
        if risk_per_unit == 0:
            return 0.0
        qty = risk_per_trade / risk_per_unit
        notional = qty * entry
        notional_cap_pct = self.cfg.get("risk", {}).get("vol_targeting", {}).get("per_symbol_notional_cap_pct_equity", 1.0)
        notional_cap = equity_usd * notional_cap_pct
        if notional > notional_cap:
            qty = notional_cap / entry
        lev_cap_qty = (equity_usd * max_leverage) / entry
        if qty > lev_cap_qty:
            qty = lev_cap_qty
        return qty
