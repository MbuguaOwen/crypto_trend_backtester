from __future__ import annotations
import os, argparse, yaml
from typing import Dict, List, Optional
import pandas as pd
from tqdm.auto import tqdm

from .data_ingest import load_1m_df
from core_reuse.utils import rolling_atr
from core_reuse.regime import TSMOMRegime
from core_reuse.trigger import BreakoutAfterCompression
from core_reuse.risk import RiskManager
from core_reuse.trade import Trade, EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL
from core_reuse.execution_helpers import ensure_min_qty

def _resample_timeframes(df1m: pd.DataFrame, tfs: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for tf in tfs:
        if tf in ("1m","1min","1minute"):
            out["1min"] = df1m.copy()
        else:
            rule = tf.replace("min","T").replace("1h","1H").replace("h","H")
            o = df1m["open"].resample(rule).first()
            h = df1m["high"].resample(rule).max()
            l = df1m["low"].resample(rule).min()
            c = df1m["close"].resample(rule).last()
            v = df1m["volume"].resample(rule).sum()
            out[tf] = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    return out

class ParityEngine:
    def __init__(self, cfg: dict, symbol: str, equity_usd: float, progress: bool=True):
        self.cfg = cfg
        self.symbol = symbol
        self.equity_usd = float(equity_usd)
        self.regime = TSMOMRegime(cfg)
        self.trigger = BreakoutAfterCompression(cfg)
        self.risk = RiskManager(cfg.get("strategy",{}).get("risk", cfg.get("risk",{})))
        self.ex_constraints = cfg.get("exchange",{}).get("symbols",{}).get(symbol,{})
        self.open_trade: Optional[Trade] = None
        self.rows = []
        self.progress = progress

    def step(self, ts, bar, df_hist):
        # Regime inputs via multi-tf (no look-ahead)
        tfs = self.cfg.get("strategy",{}).get("ts_mom",{}).get("timeframes", ["1min","5min","15min","1h"])
        tf_dfs = _resample_timeframes(df_hist, tfs)
        closes_by_tf = {k: v["close"] for k,v in tf_dfs.items() if len(v)>0}
        regime = self.regime.classify(closes_by_tf)

        atr_win = int(self.cfg.get("strategy",{}).get("risk",{}).get("atr_window", 20))
        atr_series = rolling_atr(df_hist, atr_win)
        atr_last = float(atr_series.shift(1).iloc[-1]) if len(atr_series)>0 else float("nan")

        # Update open trade & check exits
        if self.open_trade is not None:
            self.open_trade.update_levels(
                bar["close"],
                be_frac=self.risk.be_frac,
                atr=atr_last if pd.notna(atr_last) else 0.0,
                tsl_mult=self.risk.tsl_mult
            )
            if self.open_trade.check_exit(bar["high"], bar["low"]):
                self.open_trade.ts_close = ts
                pnl = (self.open_trade.exit_price - self.open_trade.entry) * self.open_trade.qty if self.open_trade.side=="LONG" else (self.open_trade.entry - self.open_trade.exit_price) * self.open_trade.qty
                self.rows.append({
                    "ts_open": self.open_trade.ts_open.isoformat(),
                    "ts_close": ts.isoformat(),
                    "symbol": self.symbol,
                    "side": self.open_trade.side,
                    "entry": round(self.open_trade.entry, 6),
                    "exit": round(float(self.open_trade.exit_price), 6),
                    "qty": round(self.open_trade.qty, 8),
                    "pnl": round(float(pnl), 6),
                    "exit_type": self.open_trade.exit_type,
                    "entry_reason": self.open_trade.entry_reason,
                })
                self.open_trade = None
                return

        # If flat, check entry
        if self.open_trade is None:
            ok, side, reason = self.trigger.check(df_hist, atr_last, regime)
            if ok and side in ("LONG","SHORT"):
                entry = bar["close"]
                sl, tp = self.risk.initial_levels(entry, side, atr_last if pd.notna(atr_last) else 0.0)
                qty = self.risk.position_size_units(
                    equity_usd=self.equity_usd,
                    entry=entry,
                    sl=sl,
                    taker_fee_bps=self.risk.taker_fee_bps,
                    side=side,
                    max_leverage=self.risk.max_leverage
                )
                amt_step = float(self.ex_constraints.get("amount_step", 0.0001))
                min_qty = float(self.ex_constraints.get("min_qty", 0.0001))
                min_notional = float(self.ex_constraints.get("min_notional", 5.0))
                qty, valid = ensure_min_qty(qty, entry, amount_step=amt_step, min_qty=min_qty, min_notional=min_notional)
                if not valid or qty<=0:
                    return
                self.open_trade = Trade(
                    symbol=self.symbol, side=side, ts_open=ts, entry=float(entry),
                    qty=float(qty), sl=float(sl), tp=float(tp), atr=float(atr_last if pd.notna(atr_last) else 0.0),
                    entry_reason=reason
                )

    def run(self, df1m: pd.DataFrame, out_csv: str):
        it = enumerate(df1m.iterrows(), 1)
        pbar = tqdm(total=len(df1m), desc=f"{self.symbol}", unit="bar", leave=False) if self.progress else None
        hist = pd.DataFrame(columns=df1m.columns)
        for i, (ts, bar) in it:
            hist = pd.concat([hist, pd.DataFrame([bar], index=[ts])], axis=0)
            if i>100:
                self.step(ts, bar, hist)
            if pbar:
                pbar.update(1)
        if pbar:
            pbar.close()
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(self.rows, columns=["ts_open","ts_close","symbol","side","entry","exit","qty","pnl","exit_type","entry_reason"]).to_csv(out_csv, index=False)

def parse_args():
    ap = argparse.ArgumentParser(description="TSMOM Parity Backtest")
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--data", required=True, help="CSV file path (1m OHLCV or ticks)")
    ap.add_argument("--out", required=True, help="Output trades CSV")
    ap.add_argument("--equity_usd", required=True, type=float)
    ap.add_argument("--no-progress", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    df1m = load_1m_df(args.data)
    eng = ParityEngine(cfg, args.symbol, args.equity_usd, progress=not args.no_progress)
    eng.run(df1m, args.out)

if __name__ == "__main__":
    main()
