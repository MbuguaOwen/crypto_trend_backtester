from __future__ import annotations
from dataclasses import dataclass

EXIT_SL = "SL"
EXIT_TP = "TP"
EXIT_BE = "BE"
EXIT_TSL = "TSL"

@dataclass
class Trade:
    symbol: str
    side: str           # LONG or SHORT
    ts_open: object
    entry: float
    qty: float
    sl: float
    tp: float
    atr: float
    entry_reason: str
    tsl: float|None = None
    be_active: bool = False
    exit_type: str|None = None
    ts_close: object|None = None
    exit_price: float|None = None

    def update_levels(self, close_price: float, *, be_frac: float, atr: float, tsl_mult: float):
        if self.side=="LONG":
            target_move = (self.tp - self.entry) * be_frac
            if (close_price - self.entry) >= target_move:
                self.sl = max(self.sl, self.entry)
                self.be_active = True
            self.tsl = close_price - tsl_mult * atr
            if self.tsl is not None:
                self.tsl = min(self.tsl, self.tp)
        else:
            target_move = (self.entry - self.tp) * be_frac
            if (self.entry - close_price) >= target_move:
                self.sl = min(self.sl, self.entry)
                self.be_active = True
            self.tsl = close_price + tsl_mult * atr
            if self.tsl is not None:
                self.tsl = max(self.tsl, self.tp)

    def check_exit(self, bar_high: float, bar_low: float):
        if self.side=="LONG":
            stop_level = self.sl
            stop_label = EXIT_BE if self.be_active and self.sl==self.entry else EXIT_SL
            if self.tsl is not None:
                if bar_low <= self.tsl:
                    self.exit_type = EXIT_TSL
                    self.exit_price = self.tsl
                    return True
                stop_level = max(stop_level, self.tsl)
                if stop_level == self.tsl:
                    stop_label = EXIT_TSL
            if bar_low <= stop_level:
                self.exit_type = stop_label
                self.exit_price = stop_level
                return True
            if bar_high >= self.tp:
                self.exit_type = EXIT_TP
                self.exit_price = self.tp
                return True
        else:
            stop_level = self.sl
            stop_label = EXIT_BE if self.be_active and self.sl==self.entry else EXIT_SL
            if self.tsl is not None:
                if bar_high >= self.tsl:
                    self.exit_type = EXIT_TSL
                    self.exit_price = self.tsl
                    return True
                stop_level = min(stop_level, self.tsl)
                if stop_level == self.tsl:
                    stop_label = EXIT_TSL
            if bar_high >= stop_level:
                self.exit_type = stop_label
                self.exit_price = stop_level
                return True
            if bar_low <= self.tp:
                self.exit_type = EXIT_TP
                self.exit_price = self.tp
                return True
        return False
