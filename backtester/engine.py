from __future__ import annotations

import os, math, json
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Tuple

from .indicators import (
    atr, bb_width_percentile, ksigma_levels, donchian_channels,
    realized_vol_ewma, tsmom_signal
)
from .risk import vol_target_notional
from .agg import TickBarAggregator
from .stream import TickCSVStream, MultiSymbolMerger

@dataclass
class Position:
    side: int = 0
    qty: float = 0.0
    entry_price: float = 0.0
    atr_at_entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    reason: str = ""
    active: bool = False
    trail_ref: float = 0.0
    moved_to_be: bool = False

class BacktestEngine:
    def __init__(self, cfg: Dict[str,Any], overrides: Optional[Dict[str,Any]] = None):
        self.cfg = cfg
        self.overrides = overrides or {}

    def _apply_overrides(self, local: Dict[str,Any]):
        if "donchian_n" in self.overrides:
            local["signals"]["donchian_n"] = self.overrides["donchian_n"]
        if "compression_percentile" in self.overrides:
            local["signals"]["compression_percentile"] = self.overrides["compression_percentile"]
        if "atr_sl_mult" in self.overrides:
            local["risk_mgmt"]["atr_sl_mult"] = self.overrides["atr_sl_mult"]
        return local

    def _iter_global_ticks_streaming(self, cfg, data_dir: str, universe_specs: list, start_ts, end_ts):
        streams = []
        chunksize = cfg.get("streaming",{}).get("chunksize", 1_000_000)
        for spec in universe_specs:
            streams.append(TickCSVStream(
                os.path.join(data_dir, spec["filename"]),
                spec["symbol"],
                chunksize=chunksize,
                start_ts=start_ts, end_ts=end_ts
            ))
        merger = MultiSymbolMerger(streams)
        for row in merger:
            yield row

    def run(
        self,
        symbols_data: List,
        outdir: Optional[str] = None,
        streaming_ctx: dict | None = None,
        tick_pbar=None,
        bar_pbar=None,
        progress_stride: int = 10_000
    ):
        cfg = self._apply_overrides(self.cfg.copy())
        bar_minutes = cfg.get("bar_interval_minutes", 1)

        equity_curve, all_trades, all_fills = [], [], []

        symbols = [spec["symbol"] for spec in (streaming_ctx["universe"] if streaming_ctx else [])] \
                  or [s for s,_ in symbols_data]
        eq0 = cfg["risk"]["equity_start"]
        per_symbol_equity = {sym: eq0 / max(1, len(symbols)) for sym in symbols}

        states = {sym: Position() for sym in symbols}
        aggs = {sym: TickBarAggregator(bar_minutes=bar_minutes) for sym in symbols}
        bars = {sym: pd.DataFrame(columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
                for sym in symbols}
        ind = {sym: {} for sym in symbols}

        streaming = cfg.get("streaming", {})
        equity_mode = streaming.get("equity_recording", "bar_close")
        equity_interval_ticks = int(streaming.get("equity_interval_ticks", 100000))
        entry_on_bar_close = bool(cfg.get("fast", {}).get("entry_on_bar_close", False))
        checkpoint_stride = int(streaming_ctx.get("checkpoint_stride_bars", 500)) if streaming_ctx else 500

        start_ts = streaming_ctx.get("start_ts") if streaming_ctx else None
        end_ts   = streaming_ctx.get("end_ts")   if streaming_ctx else None

        if streaming_ctx is not None and streaming.get("enabled", True):
            ticks_iter = self._iter_global_ticks_streaming(cfg, streaming_ctx["data_dir"], streaming_ctx["universe"],
                                                           start_ts, end_ts)
        else:
            rows = []
            for sym, df in symbols_data:
                tmp = df.copy(); tmp["symbol"] = sym
                rows.append(tmp[["timestamp","symbol","price","quantity"]])
            ticks = pd.concat(rows, axis=0, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
            def gen():
                for _, r in ticks.iterrows():
                    yield {"timestamp": pd.to_datetime(r["timestamp"], utc=True),
                           "symbol": r["symbol"], "price": float(r["price"]), "quantity": float(r["quantity"])}
            ticks_iter = gen()

        def recompute_indicators(sym: str):
            df = bars[sym]
            need = max(
                cfg["signals"]["compression_bb_n"],
                cfg["signals"]["k_sigma_lookback"],
                cfg["signals"]["donchian_n"],
                max(cfg["signals"]["tsmom_lookbacks"])
            )
            if len(df) < need + 2:
                return
            close = df["close"]; high, low = df["high"], df["low"]
            ind[sym]["atr"] = atr(high, low, close, n=cfg["risk_mgmt"]["atr_n"])
            ind[sym]["bb_width_pct"] = bb_width_percentile(close, n=cfg["signals"]["compression_bb_n"],
                                                           pct_lookback=cfg["signals"]["compression_bb_n"])
            ks = ksigma_levels(close, n=cfg["signals"]["k_sigma_lookback"])
            ind[sym]["ksigma_mu"] = ks["mu"]; ind[sym]["ksigma_sd"] = ks["sd"]
            dc = donchian_channels(high, low, n=cfg["signals"]["donchian_n"])
            ind[sym]["donch_upper"] = dc["upper"]; ind[sym]["donch_lower"] = dc["lower"]
            rets = close.pct_change().fillna(0.0)
            ind[sym]["vol"] = realized_vol_ewma(rets, lam=0.94, bar_minutes=cfg.get("bar_interval_minutes",1))
            ind[sym]["tsmom"] = tsmom_signal(close, lookbacks=cfg["signals"]["tsmom_lookbacks"],
                                             consensus_min_abs=cfg["signals"]["tsmom_consensus_min_abs"])

        def last_ind(sym: str, key: str):
            s = ind[sym].get(key)
            if s is None or len(s)==0: return np.nan
            return s.iloc[-1]

        tick_counter = 0
        processed_since_update = 0
        bars_completed = 0

        def maybe_checkpoint(ts_last: pd.Timestamp):
            if outdir and checkpoint_stride > 0 and (bars_completed % checkpoint_stride == 0):
                with open(os.path.join(outdir, "progress.json"), "w", encoding="utf-8") as f:
                    json.dump({"last_bar_ts": ts_last.isoformat(), "bars_completed": bars_completed}, f)

        for row in ticks_iter:
            ts = pd.to_datetime(row["timestamp"], utc=True)
            sym = row["symbol"]; price = float(row["price"]); qty = float(row["quantity"])

            tick_counter += 1
            processed_since_update += 1
            if tick_pbar is not None and processed_since_update >= progress_stride:
                tick_pbar.update(processed_since_update); processed_since_update = 0

            completed = aggs[sym].update(ts, price, qty)
            if completed is not None:
                # bar progressed
                bars_completed += 1
                if bar_pbar is not None: bar_pbar.update(1)

                b = completed
                bars_sym = bars[sym]
                bars_sym.loc[b.timestamp] = [b.open, b.high, b.low, b.close, b.volume]
                bars[sym] = bars_sym
                recompute_indicators(sym)

                # checkpoint every N bars
                maybe_checkpoint(b.timestamp)

                if equity_mode == "bar_close":
                    equity_curve.append({"timestamp": ts, "portfolio_equity": sum(per_symbol_equity.values())})

                # optional: entries only when bar completes (fast path)
                if entry_on_bar_close:
                    self._try_entry_manage(sym, price, ts, cfg, ind, states, per_symbol_equity)

            # default: still manage/enter intra-bar if fast mode is off
            if not entry_on_bar_close:
                self._try_entry_manage(sym, price, ts, cfg, ind, states, per_symbol_equity)

            if equity_mode != "bar_close" and (tick_counter % equity_interval_ticks == 0):
                equity_curve.append({"timestamp": ts, "portfolio_equity": sum(per_symbol_equity.values())})

        if tick_pbar is not None and processed_since_update:
            tick_pbar.update(processed_since_update)

        return {
            "equity": pd.DataFrame(equity_curve),
            "trades": pd.DataFrame([]),
            "fills":  pd.DataFrame(columns=["timestamp","symbol","side","price","qty","fee_bps","slip_bps"])
        }

    # --- entry/management logic factored out for clarity/speed ---
    def _try_entry_manage(self, sym, price, ts, cfg, ind, states, per_sym_eq):
        s = states[sym]
        # need at least one bar to have indicators
        if "tsmom" not in ind[sym] or ind[sym]["tsmom"].empty:
            return
        regime_val = self._last(ind, sym, "tsmom")
        side_regime = int(np.sign(regime_val)) if not np.isnan(regime_val) else 0
        bbp = self._last(ind, sym, "bb_width_pct")
        is_compressed = (bbp <= cfg["signals"]["compression_percentile"]) if not np.isnan(bbp) else False

        trigger = 0
        if cfg["signals"]["breakout_mode"] == "donchian":
            upper = self._last(ind, sym, "donch_upper"); lower = self._last(ind, sym, "donch_lower")
            if not np.isnan(upper) and price > upper: trigger = +1
            elif not np.isnan(lower) and price < lower: trigger = -1
        else:
            mu = self._last(ind, sym, "ksigma_mu"); sd = self._last(ind, sym, "ksigma_sd"); k = cfg["signals"]["k_sigma_k"]
            if not (np.isnan(mu) or np.isnan(sd) or sd <= 0):
                if price > mu + k*sd: trigger = +1
                elif price < mu - k*sd: trigger = -1

        # ENTRY
        if not s.active and side_regime != 0 and trigger == side_regime and is_compressed:
            vol_forecast = self._last(ind, sym, "vol")
            if np.isnan(vol_forecast) or vol_forecast <= 0: return
            sym_eq = per_sym_eq[sym]; target_vol = cfg["risk"]["target_annual_vol"]
            notional = vol_target_notional(sym_eq, target_vol, vol_forecast)
            qty_pos = max(notional / price, 0.0)
            if qty_pos <= 0: return
            atr_now = self._last(ind, sym, "atr")
            if np.isnan(atr_now) or atr_now <= 0: return
            sl_mult = cfg["risk_mgmt"]["atr_sl_mult"]; tp_mult = cfg["risk_mgmt"]["atr_tp_mult"]
            if side_regime > 0:
                s.sl = price - sl_mult * atr_now; s.tp = price + tp_mult * atr_now
            else:
                s.sl = price + sl_mult * atr_now; s.tp = price - tp_mult * atr_now
            s.side = side_regime; s.qty = qty_pos; s.entry_price = price
            s.atr_at_entry = atr_now; s.trail_ref = price; s.moved_to_be = False; s.active = True
            return

        # MANAGEMENT
        if s.active:
            hit_tp = (price >= s.tp) if s.side>0 else (price <= s.tp)
            hit_sl = (price <= s.sl) if s.side>0 else (price >= s.sl)

            progress_frac = abs((price - s.entry_price) / (s.tp - s.entry_price)) if s.tp != s.entry_price else 0.0
            if not s.moved_to_be and progress_frac >= cfg["risk_mgmt"]["breakeven_progress_frac"]:
                s.sl = s.entry_price; s.moved_to_be = True

            if cfg["risk_mgmt"].get("use_trailing", True):
                atr_now = self._last(ind, sym, "atr")
                if not (np.isnan(atr_now)) and atr_now > 0:
                    trail = cfg["risk_mgmt"]["trail_atr_mult"] * atr_now
                    if s.side > 0:
                        s.trail_ref = max(s.trail_ref, price); s.sl = max(s.sl, s.trail_ref - trail)
                    else:
                        s.trail_ref = min(s.trail_ref, price); s.sl = min(s.sl, s.trail_ref + trail)

            if hit_tp or hit_sl:
                pnl = (price - s.entry_price) * s.qty * s.side
                per_sym_eq[sym] += pnl
                states[sym] = Position()

    def _last(self, ind, sym, key):
        s = ind[sym].get(key)
        if s is None or len(s)==0: return np.nan
        return s.iloc[-1]
