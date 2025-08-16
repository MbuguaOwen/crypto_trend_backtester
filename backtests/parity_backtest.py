import argparse
import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd

import yaml

from regime import TSMOMRegime
from trigger import BreakoutAfterCompression
from risk import RiskManager
from trade import Trade, EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL
from utils import utc_ms_now, ensure_min_qty


# ---------- Data helpers ----------
def load_1m_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" in df.columns:
        idx = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.index = idx
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def downsample(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    o = df1m["open"].resample(tf).first()
    h = df1m["high"].resample(tf).max()
    l = df1m["low"].resample(tf).min()
    c = df1m["close"].resample(tf).last()
    v = df1m["volume"].resample(tf).sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    return out


def make_tf_data(df1m: pd.DataFrame, cfg: dict) -> Dict[str, pd.DataFrame]:
    tcfg = cfg["strategy"]["tsmom_regime"]["timeframes"]
    tf_data = {}
    for tf, _tc in tcfg.items():
        rule = tf.replace("m", "T").replace("h", "H").upper()
        tf_data[tf] = downsample(df1m, rule)
    return tf_data


# ---------- Exchange constraint adapter ----------
def ensure_min_qty_like_ccxt(qty: float, step: float, min_amount: float, min_cost: float, last_price: float) -> float:
    return ensure_min_qty(qty, step, min_amount, min_cost, last_price)


# ---------- Parity backtester ----------
class ParityBacktester:
    def __init__(self, cfg: dict, equity_usd: float, rules: dict):
        self.cfg = cfg
        self.equity_usd = equity_usd
        self.rules = rules
        self.regime = TSMOMRegime(cfg)
        self.trigger = BreakoutAfterCompression(cfg)
        self.risk = RiskManager(cfg)
        self.state = {}
        self.trades = []

    def _atr_last(self, df1m: pd.DataFrame) -> float:
        a = self.risk.compute_atr(df1m)
        return float(a.iloc[-1]) if a.shape[0] else 0.0

    def _maybe_entry(self, symbol: str, df1m: pd.DataFrame, tf_data: Dict[str, pd.DataFrame], now_ts: int):
        st = self.state[symbol]
        if st["open_trade"] is not None:
            return
        regime = self.regime.classify({k: v for k, v in tf_data.items()})
        sig = self.trigger.check(df1m, regime)
        if not sig:
            return
        direction, reason = sig["direction"], sig["reason"]
        price = float(df1m["close"].iloc[-1])
        atr_val = self._atr_last(df1m)
        side = "long" if direction == "long" else "short"
        symcfg = self.cfg["symbols"].get(symbol, {}) if "symbols" in self.cfg else {}
        sl, tp = self.risk.initial_levels(side, price, atr_val, symcfg)
        fees_bps = float(self.cfg.get("exchange", {}).get("taker_fee_bps", 7.5))
        max_lev = float(self.cfg.get("risk", {}).get("vol_targeting", {}).get("sizing", {}).get("max_leverage", 1.0))
        qty = self.risk.position_size_units(self.equity_usd, price, sl, fees_bps, side, max_lev)
        r = self.rules.get(symbol, {})
        step = float(r.get("amount_step", 0.0))
        min_amount = float(r.get("min_amount", 0.0))
        min_cost = float(r.get("min_cost", 0.0))
        qty = ensure_min_qty_like_ccxt(qty, step, min_amount, min_cost, price)
        if qty <= 0:
            return
        tr = Trade(symbol=symbol, side=side, entry_price=price, qty=qty, sl=sl, tp=tp,
                   ts_open=now_ts, meta={"entry_reason": reason, "initial_sl": sl})
        st["open_trade"] = tr
        st["highs_since_entry"] = float(df1m["high"].iloc[-1])
        st["lows_since_entry"] = float(df1m["low"].iloc[-1])

    def _maybe_exit(self, symbol: str, df1m: pd.DataFrame, now_ts: int):
        st = self.state[symbol]
        tr: Trade = st["open_trade"]
        if tr is None:
            return
        high = float(df1m["high"].iloc[-1])
        low = float(df1m["low"].iloc[-1])
        last = float(df1m["close"].iloc[-1])
        atr_val = self._atr_last(df1m)
        st["highs_since_entry"] = max(st["highs_since_entry"], high)
        st["lows_since_entry"] = min(st["lows_since_entry"], low)
        tr.update_levels(last_price=last, atr_val=atr_val,
                         highs_since_entry=st["highs_since_entry"],
                         lows_since_entry=st["lows_since_entry"], cfg=self.cfg)
        exit_type = tr.check_exit(high=high, low=low)
        if exit_type:
            exit_px = tr.sl if exit_type in (EXIT_SL, EXIT_BE, EXIT_TSL) else tr.tp
            pnl = (exit_px - tr.entry_price) * tr.qty if tr.side == "long" else (tr.entry_price - exit_px) * tr.qty
            self.trades.append({
                "symbol": tr.symbol,
                "side": tr.side,
                "entry": tr.entry_price,
                "exit": exit_px,
                "qty": tr.qty,
                "ts_open": tr.ts_open,
                "ts_close": now_ts,
                "exit_type": exit_type,
                "entry_reason": tr.meta.get("entry_reason", ""),
                "pnl": pnl,
            })
            st["open_trade"] = None
            st["highs_since_entry"] = -1e18
            st["lows_since_entry"] = 1e18

    def run_symbol(self, symbol: str, df1m: pd.DataFrame):
        self.state[symbol] = {"open_trade": None, "highs_since_entry": -1e18, "lows_since_entry": 1e18}
        tf_data = make_tf_data(df1m, self.cfg)
        for ts, row in df1m.iterrows():
            df_hist = df1m.loc[:ts]
            tf_hist = {tf: tdf.loc[:ts] for tf, tdf in tf_data.items()}
            if tf_hist and all(len(x) >= 5 for x in tf_hist.values()):
                self._maybe_entry(symbol, df_hist, tf_hist, int(ts.value // 1_000_000))
            self._maybe_exit(symbol, df_hist, int(ts.value // 1_000_000))

    def to_csv(self, path: str):
        pd.DataFrame(self.trades).to_csv(path, index=False)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", required=True)
    pa.add_argument("--symbol", required=True)
    pa.add_argument("--data_1m_csv", required=True)
    pa.add_argument("--out", default="trades_parity.csv")
    pa.add_argument("--equity_usd", type=float, default=10_000)
    args = pa.parse_args()

    if str(args.config).endswith(('.yaml', '.yml')):
        cfg = yaml.safe_load(open(args.config))
    else:
        cfg = json.load(open(args.config, "r"))
    df1m = load_1m_csv(args.data_1m_csv)
    rules = (cfg.get("exchange_rules", {}) or {}).get(args.symbol, {
        "amount_step": cfg.get("exchange", {}).get("amount_step", 0.0),
        "min_amount": cfg.get("exchange", {}).get("min_amount", 0.0),
        "min_cost": cfg.get("exchange", {}).get("min_cost", 0.0),
    })
    bt = ParityBacktester(cfg, equity_usd=args.equity_usd, rules={args.symbol: rules})
    bt.run_symbol(args.symbol, df1m)
    bt.to_csv(args.out)


if __name__ == "__main__":
    main()
