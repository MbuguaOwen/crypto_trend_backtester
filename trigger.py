from typing import Dict, Optional
import pandas as pd
from risk import RiskManager


class BreakoutAfterCompression:
    """Simplified breakout trigger using Donchian channels and ATR buffer."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.risk = RiskManager(cfg)

    def check(self, df1m: pd.DataFrame, regime: Dict) -> Optional[Dict[str, str]]:
        strat_cfg = self.cfg.get("strategy", {}).get("trigger", {})
        lb = strat_cfg.get("breakout", {}).get("donchian_lookback", 3)
        buffer_mult = strat_cfg.get("breakout", {}).get("buffer_atr_mult", 0.0)
        if df1m.shape[0] < lb + 1:
            return None
        upper = df1m["high"].rolling(lb).max().shift(1)
        lower = df1m["low"].rolling(lb).min().shift(1)
        atr = self.risk.compute_atr(df1m).iloc[-1]
        price = df1m["close"].iloc[-1]
        up = upper.iloc[-1] + buffer_mult * atr
        dn = lower.iloc[-1] - buffer_mult * atr
        if price > up:
            return {"direction": "long", "reason": "donchian_breakout", "level": up}
        if price < dn:
            return {"direction": "short", "reason": "donchian_breakout", "level": dn}
        return None
