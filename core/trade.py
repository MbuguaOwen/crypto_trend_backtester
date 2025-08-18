# core/trade.py
from dataclasses import dataclass, field
from typing import Optional

EXIT_SL = "SL"
EXIT_TP = "TP"
EXIT_BE = "BE"
EXIT_TSL = "TSL"

@dataclass
class Trade:
    symbol: str
    side: str  # LONG/SHORT
    entry_ts: any
    entry: float
    qty: float
    initial_sl: float
    tp: float
    fee_bps: float
    risk_r_denom: float
    trail_active: bool = False
    trail_price: Optional[float] = None
    activation_r: Optional[float] = None
    highest_close: float = field(default=None)
    lowest_close: float = field(default=None)
    mfe_price: float = field(default=None)
    exit_ts: any = None
    exit_price: float = None
    exit_type: str = None
    pnl: float = 0.0

    def fees(self, price: float) -> float:
        # simple proportional fee
        return abs(price * self.qty) * (self.fee_bps / 10000.0)
