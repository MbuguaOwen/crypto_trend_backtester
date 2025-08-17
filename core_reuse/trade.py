"""Shim Trade class with BE/TSL lifecycle hooks for standalone run.
Swap with your live trade.py for exact parity semantics.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict

EXIT_SL = "SL"
EXIT_TP = "TP"
EXIT_BE = "BE"
EXIT_TSL = "TSL"

@dataclass
class Trade:
    symbol: str
    side: str           # "LONG" or "SHORT"
    entry_price: float
    qty: float
    sl: float
    tp: float
    ts_open: int
    meta: Dict

    be_active: bool = False
    tsl_active: bool = False
    tsl_level: Optional[float] = None
    high_since: float = field(default_factory=lambda: float('-inf'))
    low_since: float  = field(default_factory=lambda: float('inf'))

    def update_levels(self, last_price: float, atr_val: float, high_since_entry: float, low_since_entry: float, cfg: dict):
        self.high_since = max(self.high_since, high_since_entry)
        self.low_since  = min(self.low_since,  low_since_entry)
        stops = cfg['risk']['stops']

        # Move to BE
        r_mult = stops['move_to_breakeven']['trigger_r_multiple']
        be_offset_atr = stops['move_to_breakeven']['be_offset_atr']
        if not self.be_active:
            if self.side == "LONG":
                if (self.high_since - self.entry_price) >= r_mult * (self.entry_price - self.sl):
                    self.be_active = True
                    self.sl = max(self.sl, self.entry_price + be_offset_atr * atr_val)
            else:
                if (self.entry_price - self.low_since) >= r_mult * (self.sl - self.entry_price):
                    self.be_active = True
                    self.sl = min(self.sl, self.entry_price - be_offset_atr * atr_val)

        # Trailing stop
        trail_mult = stops['trailing'].get('trail_atr_mult_default', 3.0)
        step_atr   = stops['trailing'].get('step_atr', 0.5)
        if self.side == "LONG":
            new_tsl = self.high_since - trail_mult * atr_val
            if self.tsl_level is None or new_tsl - self.tsl_level >= step_atr * atr_val:
                self.tsl_level = new_tsl
                self.tsl_active = True
        else:
            new_tsl = self.low_since + trail_mult * atr_val
            if self.tsl_level is None or self.tsl_level - new_tsl >= step_atr * atr_val:
                self.tsl_level = new_tsl
                self.tsl_active = True

    def check_exit(self, high: float, low: float) -> Optional[str]:
        # Order of checks: TP first, then BE/TSL vs SL
        if self.side == "LONG":
            if high >= self.tp:
                return EXIT_TP
            # BE / TSL / SL checks by price crossing
            if self.tsl_active and low <= self.tsl_level:
                return EXIT_TSL
            if low <= self.sl:
                return EXIT_SL if not self.be_active else EXIT_BE
        else:  # SHORT
            if low <= self.tp:
                return EXIT_TP
            if self.tsl_active and high >= self.tsl_level:
                return EXIT_TSL
            if high >= self.sl:
                return EXIT_SL if not self.be_active else EXIT_BE
        return None
