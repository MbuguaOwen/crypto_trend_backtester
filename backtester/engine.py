
import os
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from .events import BarEvent, OrderEvent
from .portfolio import Portfolio
from .execution import ExecutionSimulator, ExecutionConfig
from .utils.math import rolling_donchian

@dataclass
class StrategyConfig:
    momentum_windows: List[int]
    breakout_lookback: int
    vol_window: int
    target_vol_annual: float
    max_leverage: float

class MultiHorizonEngine:
    def __init__(self, symbol: str, bars: pd.DataFrame, strat_cfg: StrategyConfig, exec_cfg: ExecutionConfig, initial_capital: float, results_dir: str):
        """Event-driven engine for a single symbol."""
        self.symbol = symbol
        self.bars = bars.reset_index(drop=True)
        self.scfg = strat_cfg
        self.ecfg = exec_cfg
        self.results_dir = results_dir

        self.portfolio = Portfolio(cash=initial_capital)
        self.exec = ExecutionSimulator(exec_cfg)

        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        self._prepare_indicators()

    def _prepare_indicators(self):
        df = self.bars
        df["logret"] = (df["close"].astype(float)).pct_change().add(1).clip(lower=1e-12).pipe(lambda s: s.apply(lambda x: __import__("math").log(x)))
        df["rv_min"] = df["logret"].rolling(self.scfg.vol_window, min_periods=self.scfg.vol_window).std()
        hi = df["close"].rolling(self.scfg.breakout_lookback, min_periods=self.scfg.breakout_lookback).max()
        lo = df["close"].rolling(self.scfg.breakout_lookback, min_periods=self.scfg.breakout_lookback).min()
        df["donchian_hi"] = hi
        df["donchian_lo"] = lo
        moms = []
        for w in self.scfg.momentum_windows:
            col = f"mom_{w}"
            df[col] = df["close"].pct_change(w)
            moms.append(col)
        df["mom_score"] = df[moms].apply(lambda r: r.fillna(0.0).sum(), axis=1)
        self.bars = df

    def _warmup_bars(self) -> int:
        return max(max(self.scfg.momentum_windows), self.scfg.breakout_lookback, self.scfg.vol_window)

    def run(self):
        df = self.bars
        warmup = self._warmup_bars()

        for i in tqdm(range(len(df)), desc=f"Sim {self.symbol}", leave=False):
            row = df.iloc[i]
            ts = int(row["ts"])
            price = float(row["close"])
            vol  = float(row["volume"])

            bar = BarEvent(ts=ts, open=float(row["open"]), high=float(row["high"]), low=float(row["low"]), close=price, volume=vol, symbol=self.symbol)

            fills = self.exec.on_bar(bar_ts=ts, symbol=self.symbol, price=price)
            for f in fills:
                self.portfolio.update_fill(self.symbol, f.qty, f.price, f.fee)
                self.trades.append({"ts": f.ts, "symbol": f.symbol, "qty": f.qty, "price": f.price, "fee": f.fee})

            if i < warmup:
                self._record_equity(ts, price)
                continue

            mom = float(row["mom_score"])
            sign = 1.0 if mom > 0 else (-1.0 if mom < 0 else 0.0)

            long_ok  = price >= float(row["donchian_hi"])
            short_ok = price <= float(row["donchian_lo"])
            breakout_ok = (sign > 0 and long_ok) or (sign < 0 and short_ok)

            if sign != 0.0 and breakout_ok:
                rv_min = float(row["rv_min"]) if pd.notna(row["rv_min"]) else 0.0
                if rv_min <= 0.0:
                    target_notional = 0.0
                else:
                    from .utils.time import annualize_vol
                    curr_equity = self.portfolio.equity({self.symbol: price})
                    vol_scale = min(self.scfg.max_leverage, (self.scfg.target_vol_annual / annualize_vol(rv_min)))
                    target_notional = curr_equity * vol_scale * (1 if sign > 0 else -1)
                target_qty = target_notional / price if price > 0 else 0.0
            else:
                target_qty = 0.0

            current_qty = self.portfolio.positions.get(self.symbol, None).qty if self.symbol in self.portfolio.positions else 0.0
            delta = target_qty - current_qty
            if abs(delta) > 1e-9:
                order = OrderEvent(ts=ts, symbol=self.symbol, qty=delta, reason="rebalance")
                self.exec.on_order(order)

            self._record_equity(ts, price)

        if len(self.exec._pending) > 0 and len(df) > 0:
            final_ts = int(df.iloc[-1]["ts"]) + int(self.exec.cfg.latency_ms)
            fills = self.exec.on_bar(bar_ts=final_ts, symbol=self.symbol, price=float(df.iloc[-1]["close"]))
            for f in fills:
                self.portfolio.update_fill(self.symbol, f.qty, f.price, f.fee)
                self.trades.append({"ts": f.ts, "symbol": f.symbol, "qty": f.qty, "price": f.price, "fee": f.fee})

        os.makedirs(self.results_dir, exist_ok=True)
        if len(self.trades) > 0:
            pd.DataFrame(self.trades).to_csv(os.path.join(self.results_dir, f"{self.symbol}_trades.csv"), index=False)
        pd.DataFrame(self.equity_curve).to_csv(os.path.join(self.results_dir, f"{self.symbol}_equity.csv"), index=False)

    def _record_equity(self, ts: int, price: float):
        eq = self.portfolio.equity({self.symbol: price})
        self.equity_curve.append({"ts": ts, "equity": eq})
