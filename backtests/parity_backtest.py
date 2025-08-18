#!/usr/bin/env python3
r"""TSMOM Parity Backtest — Live 1:1 Mirror (Harness)

This harness orchestrates live modules over historical 1m bars with strict no-peek semantics.
Swap `core_reuse/*.py` with your live modules to achieve *exact* parity.
"""
from __future__ import annotations

import os, argparse, json, math
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd
from tqdm.auto import tqdm

# --- config ---
def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- imports of live-like modules (shims here) ---
from core_reuse.regime import TSMOMRegime
from core_reuse.trigger import BreakoutAfterCompression
from core_reuse.risk import RiskManager
from core_reuse.trade import Trade, EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL
from core_reuse.execution_helpers import ensure_min_qty_like_live, SymbolConstraints
from core_reuse import utils

from backtests.data_ingest import load_1m_df_for_range

def warmup_bars_required(cfg: dict) -> int:
    st = cfg['strategy']
    # regime lookbacks (in closes count; in their own tf)
    regime_look = max(v['lookback_closes'] for _, v in st['tsmom_regime']['timeframes'].items())
    # compression / donchian / ksigma
    comp = st['trigger']['compression']
    don  = st['trigger']['breakout']
    kcfg = st['trigger']['breakout']['ksigma']
    warm = max(
        regime_look + 2,
        comp['bb_window'],
        comp['min_squeeze_bars'],
        comp['lookback_for_recent_squeeze'],
        don['donchian_lookback'] + 1,
        kcfg['mean_window'] if kcfg.get('enabled', False) else 0,
        kcfg['stdev_window'] if kcfg.get('enabled', False) else 0,
        cfg['risk']['atr']['window'] + 1 + cfg['risk']['atr']['ema_halflife_bars']
    )
    return int(warm)

def _resample_tf(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf.endswith('m'):
        n = int(tf[:-1])
        rule = f"{n}min"
    elif tf.endswith('h'):
        n = int(tf[:-1])
        rule = f"{n}h"
    else:
        raise ValueError(f"Unsupported timeframe {tf}")
    o = df1m['open'].resample(rule).first()
    h = df1m['high'].resample(rule).max()
    l = df1m['low'].resample(rule).min()
    c = df1m['close'].resample(rule).last()
    v = df1m['volume'].resample(rule).sum() if 'volume' in df1m.columns else None
    out = {'open': o, 'high': h, 'low': l, 'close': c}
    if v is not None: out['volume'] = v
    return pd.DataFrame(out).dropna(how='any')

@dataclass
class TradeRow:
    ts_open: int
    ts_close: int
    symbol: str
    side: str
    entry: float
    exit: float
    qty: float
    pnl: float
    exit_type: str
    entry_reason: str

def _symbol_constraints_from_cfg(cfg: dict, symbol: str) -> SymbolConstraints:
    sc = cfg['symbols'][symbol]['precision']
    min_notional = cfg['symbols'][symbol]['min_notional_usd']
    return SymbolConstraints(min_qty=float(sc['min_qty']), step_size=float(sc['step_size']), min_notional_usd=float(min_notional))

def _apply_entry_fill(cfg: dict, df1m: pd.DataFrame, i: int) -> Tuple[int, float]:
    sim = cfg['backtest']['simulator']
    # i is the *current* completed bar index
    if sim['entry_fill'] == 'close':
        ts = int(df1m.index[i].timestamp() * 1000)
        price = float(df1m['close'].iloc[i])
        return ts, price
    else:
        # next_open
        if i + 1 >= len(df1m):
            return int(df1m.index[i].timestamp() * 1000), float(df1m['close'].iloc[i])
        ts = int(df1m.index[i+1].timestamp() * 1000)
        price = float(df1m['open'].iloc[i+1])
        return ts, price

def run_symbol(cfg: dict, symbol: str, out_path: str) -> pd.DataFrame:
    df1m = load_1m_df_for_range(symbol, cfg)
    warm = warmup_bars_required(cfg)

    # build resampled caches
    tf_defs = cfg['strategy']['tsmom_regime']['timeframes']
    # precompute resamples at full length; we'll slice per bar
    resampled = {tf: _resample_tf(df1m, tf) for tf in tf_defs.keys()}

    regime = TSMOMRegime(cfg)
    trigger = BreakoutAfterCompression(cfg)
    risk = RiskManager(cfg)

    equity_usd = cfg['account']['equity_usd']
    taker_bps  = cfg['exchange']['taker_fee_bps']
    max_lev    = cfg['risk']['vol_targeting']['sizing']['max_leverage']
    constraints = _symbol_constraints_from_cfg(cfg, symbol)

    open_trade: Optional[Trade] = None
    highs_since_entry: Optional[float] = None
    lows_since_entry: Optional[float]  = None
    # Bar index where entry was created; maintenance/exit runs strictly AFTER this bar.
    entry_bar_idx: Optional[int] = None

    rows: List[TradeRow] = []

    idx = df1m.index
    total = len(df1m)
    use_bar = tqdm(range(total), desc=symbol, unit='bar', disable=(not cfg['backtest']['progress_bar']))
    for i in use_bar:
        if i < warm:
            continue

        # 1) history slice up to current bar inclusive
        df_hist = df1m.iloc[:i+1]

        # 2) regime on TF slices with lookback windows
        tf_hist = {}
        for tf, meta in tf_defs.items():
            df_tf = resampled[tf]
            # align by time <= current bar
            df_tf_hist = df_tf[df_tf.index <= df_hist.index[-1]]
            tf_hist[tf] = df_tf_hist.tail(meta['lookback_closes'])
        state = regime.classify(tf_hist)

        # 3) trigger
        sig = trigger.check(df_hist, state) if state in ('LONG', 'SHORT') else None

        # 4) entries
        if open_trade is None and sig is not None:
            ts_open_ms, entry_px = _apply_entry_fill(cfg, df1m, i)
            atr_val = risk.compute_atr(df_hist[['high','low','close']], cfg['risk']['atr']['window'],
                                       smoothing=cfg['risk']['atr']['smoothing'],
                                       ema_halflife_bars=cfg['risk']['atr']['ema_halflife_bars'])
            if not (atr_val == atr_val):  # NaN check
                pass
            else:
                side = 'LONG' if sig['direction'] == 'long' else 'SHORT'
                symcfg = cfg['symbols'].get(symbol, {})
                sl, tp = risk.initial_levels(side, entry_px, atr_val, symcfg)
                qty = risk.position_size_units(equity_usd, entry_px, sl, taker_bps, side, max_lev)
                qty = ensure_min_qty_like_live(entry_px, qty, constraints)
                if qty > 0:
                    open_trade = Trade(symbol=symbol, side=side, entry_price=entry_px, qty=qty, sl=sl, tp=tp, ts_open=ts_open_ms,
                                       meta={'entry_reason': sig['reason'], 'initial_sl': sl})
                    # Remember which bar created this trade; we will defer maintenance to the *next* bar.
                    entry_bar_idx = i
                    # Seed post-entry extremes conservatively at the actual entry price,
                    # NOT the signal bar's H/L (avoids intra-bar lookahead).
                    highs_since_entry = float(entry_px)
                    lows_since_entry  = float(entry_px)

        # 5) maintenance and 6) exits — strictly AFTER the entry bar to avoid intra-bar lookahead
        if open_trade is not None and (entry_bar_idx is not None) and (i > entry_bar_idx):
            highs_since_entry = max(highs_since_entry, float(df1m['high'].iloc[i]))
            lows_since_entry  = min(lows_since_entry,  float(df1m['low'].iloc[i]))

            # Optional strict ATR timing: prior bars only, controlled by config.
            if cfg['backtest']['simulator'].get('atr_use_prior_bar_only', False):
                atr_hist = df_hist.iloc[:-1]
            else:
                atr_hist = df_hist

            atr_val = risk.compute_atr(atr_hist[['high','low','close']], cfg['risk']['atr']['window'],
                                       smoothing=cfg['risk']['atr']['smoothing'],
                                       ema_halflife_bars=cfg['risk']['atr']['ema_halflife_bars'])

            open_trade.update_levels(float(df1m['close'].iloc[i]), atr_val, highs_since_entry, lows_since_entry, cfg)

            exit_type = open_trade.check_exit(float(df1m['high'].iloc[i]), float(df1m['low'].iloc[i]))
            if exit_type:
                # Safety: an exit can never occur on the same bar as entry.
                assert i > entry_bar_idx, f"Leak: exit on entry bar (i={i}, entry_bar_idx={entry_bar_idx})"

                # exit price convention (unchanged)
                if exit_type == EXIT_TP:
                    exit_px = open_trade.tp
                elif exit_type == EXIT_TSL and open_trade.tsl_active and open_trade.tsl_level is not None:
                    exit_px = open_trade.tsl_level
                elif exit_type in (EXIT_SL, EXIT_BE):
                    exit_px = open_trade.sl
                else:
                    exit_px = float(df1m['close'].iloc[i])
                ts_close = int(df1m.index[i].timestamp() * 1000)

                fee_frac = utils.bps_to_frac(cfg['exchange']['taker_fee_bps'])
                if open_trade.side == 'LONG':
                    pnl_gross = (exit_px - open_trade.entry_price) * open_trade.qty
                else:
                    pnl_gross = (open_trade.entry_price - exit_px) * open_trade.qty
                entry_fee = fee_frac * open_trade.entry_price * open_trade.qty
                exit_fee  = fee_frac * exit_px * open_trade.qty
                pnl = pnl_gross - entry_fee - exit_fee

                rows.append(TradeRow(
                    ts_open=open_trade.ts_open, ts_close=ts_close, symbol=open_trade.symbol, side=open_trade.side,
                    entry=open_trade.entry_price, exit=exit_px, qty=open_trade.qty, pnl=float(pnl),
                    exit_type=exit_type, entry_reason=open_trade.meta.get('entry_reason','')
                ))
                open_trade = None
                entry_bar_idx = None
                highs_since_entry = None
                lows_since_entry  = None

    # write CSV
    out_df = pd.DataFrame([r.__dict__ for r in rows])
    if len(out_df) == 0:
        # still write header to be deterministic
        out_df = pd.DataFrame(columns=['ts_open','ts_close','symbol','side','entry','exit','qty','pnl','exit_type','entry_reason'])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def summarize(df: pd.DataFrame, sym: str, out_dir: str = "outputs"):
    import numpy as np, os
    os.makedirs(out_dir, exist_ok=True)

    total = len(df)
    counts = df.groupby("exit_type").size().reindex(["TP","TSL","SL","BE"], fill_value=0)

    # ---- Realized statistics ----
    pos_mask = df["pnl"] > 0
    neg_mask = df["pnl"] < 0
    wins_real   = int(pos_mask.sum())
    losses_real = int(neg_mask.sum())
    winrate_real = wins_real / max(1, total)
    pnl_total = float(df["pnl"].sum()) if total else 0.0
    avg_win  = float(df.loc[pos_mask, "pnl"].mean() or 0.0)
    avg_loss = float(-df.loc[neg_mask, "pnl"].mean() or 0.0)  # positive
    expectancy = winrate_real * avg_win - (1.0 - winrate_real) * avg_loss

    summary = {
        "symbol": sym,
        "trades": int(total),
        "TP": int(counts["TP"]), "TSL": int(counts["TSL"]),
        "SL": int(counts["SL"]), "BE": int(counts["BE"]),
        "wins_real": wins_real, "losses_real": losses_real,
        "win_rate_real": round(winrate_real, 4),
        "net_pnl": round(pnl_total, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 4),
    }

    # Console log with realized win rate
    print(f"\n[{sym}] trades={summary['trades']} | TP={summary['TP']} TSL={summary['TSL']} "
          f"SL={summary['SL']} BE={summary['BE']} | win_rate_real={summary['win_rate_real']:.3f} "
          f"net_pnl={summary['net_pnl']:.2f} exp={summary['expectancy']:.4f}")

    # Save summary CSV
    import pandas as pd
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"{sym}_summary.csv"), index=False)

    # Save a per-exit breakdown for auditability
    breakdown = (df.groupby('exit_type')
                   .agg(trades=('pnl','size'),
                        wins=('pnl', lambda s: int((s>0).sum())),
                        losses=('pnl', lambda s: int((s<0).sum())),
                        mean_pnl=('pnl','mean'),
                        median_pnl=('pnl','median'),
                        sum_pnl=('pnl','sum'))
                   .reset_index())
    breakdown.to_csv(os.path.join(out_dir, f"{sym}_exit_breakdown.csv"), index=False)

    return summary

def main():
    ap = argparse.ArgumentParser(description="TSMOM Parity Backtest Harness")
    ap.add_argument('--config', default='configs/default.yaml', help='Config YAML path')
    ap.add_argument('--symbol', default=None, help='Single symbol to run; default=all in config.universe.symbols')
    ap.add_argument('--out', default=None, help='Override output CSV path')
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    symbols = [args.symbol] if args.symbol else list(cfg['universe']['symbols'])
    combined = []

    for sym in symbols:
        out_path = args.out or cfg['backtest']['output']['trades_csv'].replace('{symbol}', sym)
        df = run_symbol(cfg, sym, out_path)
        summarize(df, sym)
        df['symbol'] = sym
        combined.append(df)

    # Optionally aggregate; here we just run per-symbol

if __name__ == '__main__':
    main()
