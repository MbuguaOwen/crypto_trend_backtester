
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Position:
    qty: float = 0.0
    entry_price: float = 0.0

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def ensure(self, symbol: str):
        if symbol not in self.positions:
            self.positions[symbol] = Position()

    def update_fill(self, symbol: str, fill_qty: float, fill_price: float, fee: float):
        self.ensure(symbol)
        pos = self.positions[symbol]
        self.cash -= fill_qty * fill_price
        self.cash -= fee

        new_qty = pos.qty + fill_qty
        if abs(new_qty) < 1e-12:
            pos.qty = 0.0
            pos.entry_price = 0.0
        elif pos.qty == 0.0:
            pos.qty = new_qty
            pos.entry_price = fill_price
        else:
            pos.entry_price = (pos.entry_price * pos.qty + fill_qty * fill_price) / new_qty
            pos.qty = new_qty

    def market_value(self, prices: Dict[str, float]) -> float:
        mv = 0.0
        for s, p in self.positions.items():
            price = prices.get(s, 0.0)
            mv += p.qty * price
        return mv

    def equity(self, prices: Dict[str, float]) -> float:
        return self.cash + self.market_value(prices)
