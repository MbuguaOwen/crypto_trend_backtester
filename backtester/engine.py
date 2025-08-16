import os
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

from .events import BarEvent, OrderEvent
from .portfolio import Portfolio
from .execution import ExecutionSimulator, ExecutionConfig


@dataclass
class StrategyConfig:
    momentum_windows: List[int]
    breakout_lookback: int
    vol_window: int
    target_vol_annual: float
    max_leverage: float


@dataclass
class DebugCounters:
    total_bars: int = 0
    warmup: int = 0
    bars_post_warmup: int = 0
    mom_pos: int = 0
    mom_neg: int = 0
    breakout_long: int = 0
    breakout_short: int = 0
    breakout_ok: int = 0
    rv_nonpos: int = 0
    orders: int = 0


class MultiHorizonEngine:
    def __init__(self, symbol: str, bars: pd.DataFrame, strat_cfg: StrategyConfig,
                 exec_cfg: ExecutionConfig, initial_capital: float, results_dir: str):
        """Event-driven engine for a single symbol. bars: ['ts','open','high','low','close','volume']"""
        self.symbol = symbol
        self.bars = bars.reset_index(drop=True).copy()
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
        # returns & vol
        df["logret"] = df["close"].astype(float).pct_change().add(1).clip(lower=1e-12).pipe(
            lambda s: s.apply(lambda x: __import__("math").log(x))
        )
        df["rv_min"] = df["logret"].rolling(self.scfg.vol_window, min_periods=self.scfg.vol_window).std()
        # Donchian
        df["donchian_hi"] = df["close"].rolling(self.scfg.breakout_lookback, min_periods=self.scfg.breakout_lookback).max()
        df["donchian_lo"] = df["close"].rolling(self.scfg.breakout_lookback, min_periods=self.scfg.breakout_lookback).min()
        # Momentum composite (sum of pct changes)
        mom_cols = []
        for w in self.scfg.momentum_windows:
            col = f"mom_{w}"
            df[col] = df["close"].pct_change(w)
            mom_cols.append(col)
        df["mom_score"] = df[mom_cols].fillna(0.0).sum(axis=1)
        self.bars = df

    def _warmup_bars(self) -> int:
        return max(max(self.scfg.momentum_windows), self.scfg.breakout_lookback, self.scfg.vol_window)

    def run(self, debug: bool = False):
        df = self.bars
        warmup = self._warmup_bars()

        diag = DebugCounters(total_bars=len(df), warmup=warmup)
        debug_rows: List[Dict] = []

        for i in tqdm(range(len(df)), desc=f"Sim {self.symbol}", leave=False):
            row = df.iloc[i]
            ts = int(row["ts"])
            price = float(row["close"])
            vol  = float(row["volume"])

            # Fill pending
            fills = self.exec.on_bar(bar_ts=ts, symbol=self.symbol, price=price)
            for f in fills:
                self.portfolio.update_fill(self.symbol, f.qty, f.price, f.fee)
                self.trades.append({"ts": f.ts, "symbol": f.symbol, "qty": f.qty, "price": f.price, "fee": f.fee})

            if i < warmup:
                self._record_equity(ts, price)
                continue

            diag.bars_post_warmup += 1

            mom = float(row["mom_score"])
            sign = 1.0 if mom > 0 else (-1.0 if mom < 0 else 0.0)
            if sign > 0: diag.mom_pos += 1
            elif sign < 0: diag.mom_neg += 1

            long_ok  = price >= float(row["donchian_hi"]) if pd.notna(row["donchian_hi"]) else False
            short_ok = price <= float(row["donchian_lo"]) if pd.notna(row["donchian_lo"]) else False
            if long_ok: diag.breakout_long += 1
            if short_ok: diag.breakout_short += 1
            breakout_ok = (sign > 0 and long_ok) or (sign < 0 and short_ok)
            if breakout_ok: diag.breakout_ok += 1

            rv_min = float(row["rv_min"]) if pd.notna(row["rv_min"]) else 0.0
            if rv_min <= 0.0: diag.rv_nonpos += 1

            if sign != 0.0 and breakout_ok and rv_min > 0.0:
                # Vol targeting
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
                self.exec.on_order(OrderEvent(ts=ts, symbol=self.symbol, qty=delta, reason="rebalance"))
                diag.orders += 1

            if debug:
                debug_rows.append({
                    "ts": ts, "close": price, "mom_score": mom,
                    "donchian_hi": row.get("donchian_hi"), "donchian_lo": row.get("donchian_lo"),
                    "rv_min": rv_min, "sign": sign,
                    "long_ok": long_ok, "short_ok": short_ok, "breakout_ok": breakout_ok,
                    "target_qty": target_qty, "current_qty": current_qty, "delta": delta
                })

            self._record_equity(ts, price)

        # Final fill sweep
        if len(self.exec._pending) > 0 and len(df) > 0:
            final_ts = int(df.iloc[-1]["ts"]) + int(self.exec.cfg.latency_ms)
            fills = self.exec.on_bar(bar_ts=final_ts, symbol=self.symbol, price=float(df.iloc[-1]["close"]))
            for f in fills:
                self.portfolio.update_fill(self.symbol, f.qty, f.price, f.fee)
                self.trades.append({"ts": f.ts, "symbol": f.symbol, "qty": f.qty, "price": f.price, "fee": f.fee})

        os.makedirs(self.results_dir, exist_ok=True)
        # Persist regular outputs
        if len(self.trades) > 0:
            pd.DataFrame(self.trades).to_csv(os.path.join(self.results_dir, f"{self.symbol}_trades.csv"), index=False)
        pd.DataFrame(self.equity_curve).to_csv(os.path.join(self.results_dir, f"{self.symbol}_equity.csv"), index=False)

        # Persist diagnostics
        pd.Series({
            "total_bars": diag.total_bars,
            "warmup": diag.warmup,
            "bars_post_warmup": diag.bars_post_warmup,
            "mom_pos": diag.mom_pos,
            "mom_neg": diag.mom_neg,
            "breakout_long": diag.breakout_long,
            "breakout_short": diag.breakout_short,
            "breakout_ok": diag.breakout_ok,
            "rv_nonpos": diag.rv_nonpos,
            "orders_enqueued": diag.orders,
        }).to_json(os.path.join(self.results_dir, f"{self.symbol}_diag.json"))

        if debug and debug_rows:
            pd.DataFrame(debug_rows).to_csv(os.path.join(self.results_dir, f"{self.symbol}_debug.csv"), index=False)

    def _record_equity(self, ts: int, price: float):
        eq = self.portfolio.equity({self.symbol: price})
        self.equity_curve.append({"ts": ts, "equity": eq})
