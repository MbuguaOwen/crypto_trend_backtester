# backtester/agg.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import timezone

def floor_to_minute(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    ts = ts.tz_convert(timezone.utc) if ts.tzinfo else ts.tz_localize(timezone.utc)
    floored = ts.replace(second=0, microsecond=0, nanosecond=0)
    if minutes > 1:
        minute = (floored.minute // minutes) * minutes
        floored = floored.replace(minute=minute)
    return floored

@dataclass
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

class TickBarAggregator:
    def __init__(self, bar_minutes: int = 1):
        self.bar_minutes = bar_minutes
        self.cur_ts: pd.Timestamp | None = None
        self.open = self.high = self.low = self.close = None
        self.volume = 0.0

    def update(self, ts: pd.Timestamp, price: float, qty: float):
        bucket = floor_to_minute(ts, self.bar_minutes)
        if self.cur_ts is None:
            self.cur_ts = bucket
            self.open = self.high = self.low = self.close = price
            self.volume = float(qty)
            return None
        if bucket == self.cur_ts:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            self.close = price
            self.volume += float(qty)
            return None
        completed = Bar(self.cur_ts, self.open, self.high, self.low, self.close, self.volume)
        self.cur_ts = bucket
        self.open = self.high = self.low = self.close = price
        self.volume = float(qty)
        return completed
