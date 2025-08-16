
from dataclasses import dataclass
from typing import List
from .events import OrderEvent, FillEvent

@dataclass
class ExecutionConfig:
    taker_fee_bps: float = 7.5
    slippage_bps: float = 1.5
    latency_ms: int = 500

class ExecutionSimulator:
    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg
        self._pending: List[OrderEvent] = []

    def on_order(self, order: OrderEvent):
        self._pending.append(order)

    def on_bar(self, bar_ts: int, symbol: str, price: float):
        fills: List[FillEvent] = []
        keep: List[OrderEvent] = []
        for o in self._pending:
            if o.symbol != symbol:
                keep.append(o)
                continue
            if bar_ts - o.ts < self.cfg.latency_ms:
                keep.append(o)
                continue
            slip = price * (self.cfg.slippage_bps / 1e4)
            fill_price = price + (slip if o.qty > 0 else -slip)
            fee = abs(o.qty * fill_price) * (self.cfg.taker_fee_bps / 1e4)
            fills.append(FillEvent(ts=bar_ts, symbol=symbol, qty=o.qty, price=fill_price, fee=fee))
        self._pending = keep
        return fills
