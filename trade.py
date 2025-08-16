from dataclasses import dataclass, field
from typing import Dict, Optional

EXIT_SL = "SL"
EXIT_TP = "TP"
EXIT_BE = "BE"
EXIT_TSL = "TSL"


@dataclass
class Trade:
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    qty: float
    sl: float
    tp: float
    ts_open: int
    meta: Dict = field(default_factory=dict)

    def update_levels(self, last_price: float, atr_val: float, highs_since_entry: float, lows_since_entry: float, cfg: Dict) -> Optional[Dict]:
        stops_cfg = cfg.get("risk", {}).get("stops", {})
        be_cfg = stops_cfg.get("move_to_breakeven", {})
        trail_cfg = stops_cfg.get("trailing", {})
        # move to break-even
        r_multiple = None
        if self.side == "long":
            r_multiple = (highs_since_entry - self.entry_price) / max(self.entry_price - self.meta.get("initial_sl", self.sl), 1e-9)
            if r_multiple >= be_cfg.get("trigger_r_multiple", 1e9) and self.sl < self.entry_price:
                self.sl = self.entry_price + be_cfg.get("be_offset_atr", 0.0) * atr_val
                return {"event": "SL_TO_BE"}
            # trailing stop
            trail_mult = trail_cfg.get("trail_atr_mult_default", 0.0)
            step = trail_cfg.get("step_atr", 0.0)
            new_sl = highs_since_entry - trail_mult * atr_val
            if new_sl - self.sl >= step * atr_val:
                self.sl = new_sl
                return {"event": "TRAIL_UPDATE"}
        else:
            r_multiple = (self.entry_price - lows_since_entry) / max(self.meta.get("initial_sl", self.sl) - self.entry_price, 1e-9)
            if r_multiple >= be_cfg.get("trigger_r_multiple", 1e9) and self.sl > self.entry_price:
                self.sl = self.entry_price - be_cfg.get("be_offset_atr", 0.0) * atr_val
                return {"event": "SL_TO_BE"}
            trail_mult = trail_cfg.get("trail_atr_mult_default", 0.0)
            step = trail_cfg.get("step_atr", 0.0)
            new_sl = lows_since_entry + trail_mult * atr_val
            if self.sl - new_sl >= step * atr_val:
                self.sl = new_sl
                return {"event": "TRAIL_UPDATE"}
        return None

    def check_exit(self, high: float, low: float) -> Optional[str]:
        if self.side == "long":
            if low <= self.sl:
                if abs(self.sl - self.entry_price) < 1e-9:
                    return EXIT_BE
                elif self.sl > self.entry_price:
                    return EXIT_TSL
                else:
                    return EXIT_SL
            if high >= self.tp:
                return EXIT_TP
        else:
            if high >= self.sl:
                if abs(self.sl - self.entry_price) < 1e-9:
                    return EXIT_BE
                elif self.sl < self.entry_price:
                    return EXIT_TSL
                else:
                    return EXIT_SL
            if low <= self.tp:
                return EXIT_TP
        return None
