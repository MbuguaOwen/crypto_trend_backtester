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

        # --- Fee/slippage buffer (round trip) for BE ---
        taker_bps = float(cfg['exchange'].get('taker_fee_bps', 0.0))
        fee_frac  = taker_bps / 10000.0
        sim = cfg.get('backtest', {}).get('simulator', {})
        slip_bps = float(sim.get('slippage_bps', 0.0))
        slip_frac = slip_bps / 10000.0
        rt_cost_frac = 2.0 * (fee_frac + slip_frac)  # entry+exit
        be_fee_buffer = self.entry_price * rt_cost_frac * 1.10  # +10% cushion

        # --- Move to BE with protection against fee-loss "scratches" ---
        r_mult = float(stops.get('move_to_breakeven', {}).get('trigger_r_multiple', 1.0))
        be_offset_atr = float(stops.get('move_to_breakeven', {}).get('be_offset_atr', 0.0))

        if not self.be_active:
            if self.side == "LONG":
                # Profit in R terms relative to initial risk
                if (self.high_since - self.entry_price) >= r_mult * (self.entry_price - self.sl):
                    self.be_active = True
                    # Must at least cover fees/slippage
                    target_be = max(self.entry_price + be_offset_atr * atr_val, self.entry_price + be_fee_buffer)
                    self.sl = max(self.sl, target_be)
            else:
                if (self.entry_price - self.low_since) >= r_mult * (self.sl - self.entry_price):
                    self.be_active = True
                    target_be = min(self.entry_price - be_offset_atr * atr_val, self.entry_price - be_fee_buffer)
                    self.sl = min(self.sl, target_be)

        # --- Trailing stop: arm only after BE and/or R-multiple; floor relative to SL/BE; step in ATR units ---
        trail = stops.get('trailing', {})
        trail_mult = float(trail.get('trail_atr_mult_default', 3.0))
        step_atr   = float(trail.get('step_atr', 0.5))
        arm_after_be = bool(trail.get('activate_after_be', True))
        arm_after_r  = float(trail.get('activate_after_r_multiple', 1.0))
        min_gap_atr  = float(trail.get('min_gap_atr', 0.5))

        def _arm_ok_long():
            cond_r = (self.high_since - self.entry_price) >= arm_after_r * (self.entry_price - self.sl)
            return (self.be_active if arm_after_be else True) and cond_r

        def _arm_ok_short():
            cond_r = (self.entry_price - self.low_since) >= arm_after_r * (self.sl - self.entry_price)
            return (self.be_active if arm_after_be else True) and cond_r

        if self.side == "LONG":
            if _arm_ok_long():
                new_tsl = self.high_since - trail_mult * atr_val
                # Floor TSL so it's not worse than SL/BE plus a small ATR gap
                floor_lvl = self.sl + min_gap_atr * atr_val
                new_tsl = max(new_tsl, floor_lvl)
                if self.tsl_level is None or new_tsl - self.tsl_level >= step_atr * atr_val:
                    self.tsl_level = new_tsl
                    self.tsl_active = True
        else:
            if _arm_ok_short():
                new_tsl = self.low_since + trail_mult * atr_val
                # Floor for shorts: not worse than SL/BE minus a small ATR gap
                floor_lvl = self.sl - min_gap_atr * atr_val
                new_tsl = min(new_tsl, floor_lvl)
                if self.tsl_level is None or self.tsl_level - new_tsl >= step_atr * atr_val:
                    self.tsl_level = new_tsl
                    self.tsl_active = True

    def check_exit(self, high: float, low: float) -> Optional[str]:
        """
        Order of checks:
          - TP first (profits always realized)
          - SL BEFORE TSL to avoid misclassifying SL as TSL in collisions
          - TSL only triggers if it is tighter (i.e., not worse than SL/BE)
        """
        if self.side == "LONG":
            if high >= self.tp:
                return EXIT_TP
            # Prefer SL if both SL and TSL could be tagged on same bar, or if TSL worse than SL
            if low <= self.sl:
                return EXIT_BE if self.be_active else EXIT_SL
            if self.tsl_active and self.tsl_level is not None and self.tsl_level >= self.sl and low <= self.tsl_level:
                return EXIT_TSL
        else:  # SHORT
            if low <= self.tp:
                return EXIT_TP
            if high >= self.sl:
                return EXIT_BE if self.be_active else EXIT_SL
            if self.tsl_active and self.tsl_level is not None and self.tsl_level <= self.sl and high >= self.tsl_level:
                return EXIT_TSL
        return None
