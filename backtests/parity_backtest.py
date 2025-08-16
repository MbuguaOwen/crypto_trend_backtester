from __future__ import annotations
import os, argparse, yaml
from typing import Dict, List
from datetime import datetime

import pandas as pd
from tqdm.auto import tqdm

from .data_ingest import load_1m_df
from core_reuse.utils import rolling_atr
from core_reuse.regime import TSMOMRegime
from core_reuse.trigger import BreakoutAfterCompression
from core_reuse.risk import RiskManager
from core_reuse.trade import Trade, EXIT_SL, EXIT_TP
from core_reuse.execution_helpers import ensure_min_qty


def _normalize_freq(tf: str) -> str:
    tf = tf.strip().lower()
    if tf in {"1m", "1min", "1minute"}:
        return "1min"
    if tf.endswith("m") and tf[:-1].isdigit():
        return f"{int(tf[:-1])}min"
    if tf.endswith("min") and tf[:-3].isdigit():
        return f"{int(tf[:-3])}min"
    if tf.endswith("h") and tf[:-1].isdigit():
        return f"{int(tf[:-1])}h"
    return tf


def _resample_timeframes(df1m: pd.DataFrame, tfs: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for tf in tfs:
        rule = _normalize_freq(tf)
        if rule == "1min":
            out["1min"] = df1m
        else:
            o = df1m["open"].resample(rule).first()
            h = df1m["high"].resample(rule).max()
            l = df1m["low"].resample(rule).min()
            c = df1m["close"].resample(rule).last()
            v = df1m["volume"].resample(rule).sum()
            out[rule] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    return out


def _expand_months_to_files(template: str, symbol: str, months: List[str]) -> List[str]:
    files = []
    for m in months:
        dt = datetime.strptime(m, "%Y-%m")
        files.append(template.format(symbol=symbol, YYYY=f"{dt.year:04d}", MM=f"{dt.month:02d}"))
    return files

class ParityEngine:
    def __init__(self, cfg: dict, symbol: str, equity_usd: float, progress: bool=True):
        self.cfg = cfg
        self.symbol = symbol
        self.equity_usd = float(equity_usd)
        self.regime = TSMOMRegime(cfg)
        self.trigger = BreakoutAfterCompression(cfg)
        self.risk = RiskManager(cfg.get("strategy",{}).get("risk", cfg.get("risk",{})))
        self.ex_constraints = cfg.get("exchange",{}).get("symbols",{}).get(symbol,{})
        self.open_trade = None
        self.rows = []
        self.progress = progress

        # ---- NEW: dynamic warmup based on longest lookback ----
        trig = self.cfg.get("strategy",{}).get("trigger",{})
        comp = trig.get("compression",{})
        ks   = trig.get("ksigma",{})
        risk = self.cfg.get("strategy",{}).get("risk",{})
        self.warmup = max(
            int(trig.get("donchian_lookback", 20)),
            int(comp.get("bb_window", 50)),
            int(ks.get("window", 60)),
            int(risk.get("atr_window", 20)),
        ) + 2  # a tiny buffer

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

    def run_df(self, df1m: pd.DataFrame):
        """Process a single 1m dataframe without writing to disk."""
        it = enumerate(df1m.iterrows(), 1)
        pbar = tqdm(total=len(df1m), desc=f"{self.symbol}", unit="bar", leave=False) if self.progress else None

        for i, (ts, bar) in it:
            # Skip until we have enough history for ATR/Donchian/compression etc.
            if i <= self.warmup:
                if pbar:
                    pbar.update(1)
                continue

            # History up to and including this bar; no concat, no warnings.
            df_hist = df1m.iloc[:i]
            self.step(ts, bar, df_hist)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    def write_csv(self, out_csv: str):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(
            self.rows,
            columns=["ts_open","ts_close","symbol","side","entry","exit","qty","pnl","exit_type","entry_reason"]
        ).to_csv(out_csv, index=False)

    def run(self, df1m: pd.DataFrame, out_csv: str):
        it = enumerate(df1m.iterrows(), 1)
        pbar = tqdm(total=len(df1m), desc=f"{self.symbol}", unit="bar", leave=False) if self.progress else None

        for i, (ts, bar) in it:
            # Skip until we have enough history for ATR/Donchian/compression etc.
            if i <= self.warmup:
                if pbar:
                    pbar.update(1)
                continue

            # History up to and including this bar; no concat, no warnings.
            df_hist = df1m.iloc[:i]
            self.step(ts, bar, df_hist)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        # Final forced exit if position remains open at end of data
        if self.open_trade is not None:
            ts = df1m.index[-1]
            last_close = float(df1m.iloc[-1]["close"])
            self.open_trade.ts_close = ts
            self.open_trade.exit_price = last_close
            pnl = (last_close - self.open_trade.entry) * self.open_trade.qty if self.open_trade.side == "LONG" else (self.open_trade.entry - last_close) * self.open_trade.qty
            exit_type = EXIT_TP if (last_close >= self.open_trade.entry if self.open_trade.side == "LONG" else last_close <= self.open_trade.entry) else EXIT_SL
            self.rows.append({
                "ts_open": self.open_trade.ts_open.isoformat(),
                "ts_close": ts.isoformat(),
                "symbol": self.symbol,
                "side": self.open_trade.side,
                "entry": round(self.open_trade.entry, 6),
                "exit": round(last_close, 6),
                "qty": round(self.open_trade.qty, 8),
                "pnl": round(float(pnl), 6),
                "exit_type": exit_type,
                "entry_reason": self.open_trade.entry_reason,
            })
            self.open_trade = None

        # dump trades
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(
            self.rows,
            columns=["ts_open","ts_close","symbol","side","entry","exit","qty","pnl","exit_type","entry_reason"]
        ).to_csv(out_csv, index=False)

def parse_args():
    ap = argparse.ArgumentParser(description="TSMOM Parity Backtest")
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--data", default=None, help="CSV file path (1m OHLCV or ticks)")
    ap.add_argument("--out", required=True, help="Output trades CSV")
    ap.add_argument("--equity_usd", required=True, type=float)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--months", default=None,
                    help="Comma-separated YYYY-MM list (e.g., 2025-01,2025-02). If set, will run over multiple files using data_template/config.")
    ap.add_argument("--data-template", default=None,
                    help="Python format string with {symbol},{YYYY},{MM} (e.g., data/{symbol}/{symbol}-1m-{YYYY}-{MM}.csv). Takes precedence over config backtest.data_template.")
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_bt = (cfg.get("backtest") or {})
    # CLI has priority for single-file
    if args.data:
        # Single-file mode
        df1m = load_1m_df(args.data)
        eng = ParityEngine(cfg, args.symbol, args.equity_usd, progress=not args.no_progress)
        eng.run(df1m, args.out)
        return

    # Multi-file mode (months + template) only if --data is NOT provided
    months = []
    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    else:
        months = list(cfg_bt.get("months", []))

    # Use CLI template if given; otherwise config; otherwise default to TICKS pattern
    data_template = args.data_template or cfg_bt.get("data_template", "data/{symbol}/{symbol}-ticks-{YYYY}-{MM}.csv")

    if not months:
        raise SystemExit("No --data provided and no months configured. Set --months or backtest.months in config.yaml.")

    files = _expand_months_to_files(data_template, args.symbol, months)

    # Auto-fallback: if a built path doesn't exist and template looks like 1m, try the ticks template, and vice versa
    import os
    resolved = []
    for fp in files:
        if os.path.exists(fp):
            resolved.append(fp)
            continue
        # try common alternates
        alts = []
        if "-1m-" in fp:
            alts.append(fp.replace("-1m-", "-ticks-"))
        if "-ticks-" in fp:
            alts.append(fp.replace("-ticks-", "-1m-"))
        alt_hit = None
        for a in alts:
            if os.path.exists(a):
                alt_hit = a
                break
        if alt_hit:
            resolved.append(alt_hit)
        else:
            raise FileNotFoundError(f"Could not find {fp} (or alternates). Check backtest.data_template / filenames.")

    eng = ParityEngine(cfg, args.symbol, args.equity_usd, progress=not args.no_progress)
    parent = tqdm(resolved, desc=f"Files:{args.symbol}", unit="file", leave=True)
    for fp in parent:
        df1m = load_1m_df(fp)
        eng.run_df(df1m)
    eng.write_csv(args.out)

if __name__ == "__main__":
    main()
