
from dataclasses import dataclass

@dataclass
class BarEvent:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

@dataclass
class SignalEvent:
    ts: int
    symbol: str
    target_pos: float

@dataclass
class OrderEvent:
    ts: int
    symbol: str
    qty: float
    reason: str

@dataclass
class FillEvent:
    ts: int
    symbol: str
    qty: float
    price: float
    fee: float
