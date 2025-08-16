from __future__ import annotations

class RiskManager:
    def __init__(self, cfg: dict):
        r = cfg if "atr_window" in cfg else cfg.get("risk", cfg.get("strategy",{}).get("risk",{}))
        self.atr_window = int(r.get("atr_window", 20))
        self.atr_mult_sl = float(r.get("atr_mult_sl", 2.0))
        self.atr_mult_tp = float(r.get("atr_mult_tp", 3.0))
        self.be_frac = float(r.get("breakeven_trigger_frac_of_tp", 0.6))
        self.tsl_mult = float(r.get("trailing_atr_mult", 1.0))
        self.risk_fraction = float(r.get("risk_fraction", 0.005))
        self.taker_fee_bps = float(r.get("taker_fee_bps", 5.0))
        self.max_leverage = float(r.get("max_leverage", 5.0))

    def initial_levels(self, entry: float, side: str, atr: float):
        if side=="LONG":
            sl = entry - self.atr_mult_sl * atr
            tp = entry + self.atr_mult_tp * atr
        else:
            sl = entry + self.atr_mult_sl * atr
            tp = entry - self.atr_mult_tp * atr
        return sl, tp

    def position_size_units(self, equity_usd: float, entry: float, sl: float, taker_fee_bps: float|None, side: str, max_leverage: float|None):
        risk_dollars = float(equity_usd) * self.risk_fraction
        dist = abs(entry - sl)
        if dist <= 0:
            return 0.0
        qty = risk_dollars / dist
        if max_leverage:
            notional = qty * entry
            max_notional = float(equity_usd) * float(max_leverage)
            if notional > max_notional:
                qty = max_notional / entry
        if taker_fee_bps:
            qty *= (1.0 - float(taker_fee_bps)/10000.0)
        return max(qty, 0.0)
