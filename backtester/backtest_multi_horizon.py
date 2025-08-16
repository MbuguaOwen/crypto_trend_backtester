from __future__ import annotations

"""Event driven multi‑horizon backtester.

The implementation is intentionally compact – it does not aim to be a production
grade trading system but rather a faithful, reproducible scaffold that mirrors
live behaviour.  The module wires together the surrounding helpers in this
package and follows the blueprint provided in the task description.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from . import execution_sim, metrics, ohlc, portfolio, risk, signals, utils


def run_backtest(cfg: Dict, results_dir: str) -> None:
    utils.ensure_dir(results_dir)
    logger = utils.get_logger("backtest", os.path.join(results_dir, "backtest.log"))
    utils.seed_everything(cfg.get("seed", 42))

    all_trade_returns: List[pd.Series] = []

    for symbol in cfg["data"]["symbols"]:
        logger.info(f"Loading ticks for {symbol}")
        ticks = ohlc.load_ticks(symbol, cfg["data"]["months"], cfg["data"]["dir"])
        bars_1m = ohlc.build_ohlcv(ticks)
        bars_15m = ohlc.resample_from_1m(bars_1m, cfg["horizons"]["micro"])
        bars_60m = ohlc.resample_from_1m(bars_1m, cfg["horizons"]["macro"])
        bars_4h = ohlc.resample_from_1m(bars_1m, "240T")
        bars_1d = ohlc.resample_from_1m(bars_1m, cfg["horizons"]["daily"])

        atr_series = risk.atr(bars_1m, cfg["risk"]["atr_period"])
        don = signals.donchian(bars_1m, cfg["signals"]["trigger"]["donchian_lookback"])
        comp = signals.compression(bars_1m, cfg["signals"]["compression"])
        micro = signals.micro_slope(bars_1m.index, bars_15m, cfg["signals"]["micro"])
        macro = signals.macro_regime(bars_1m.index, bars_60m, bars_4h, bars_1d, cfg["signals"]["regime"])
        returns_1m = np.log(bars_1m["close"] / bars_1m["close"].shift(1))
        sigma = portfolio.vol_forecast(returns_1m, cfg["sizing"])

        feat = pd.DataFrame({
            "open": bars_1m["open"],
            "high": bars_1m["high"],
            "low": bars_1m["low"],
            "close": bars_1m["close"],
            "atr": atr_series,
            "don_high": don["high"],
            "don_low": don["low"],
            "compression": comp,
            "micro": micro,
            "macro": macro,
            "sigma": sigma,
        }).dropna()

        if feat.empty:
            utils.write_advisory(os.path.join(results_dir, "advisory.txt"), f"No data for {symbol}")
            continue

        t0 = feat.index[0]
        warmup_end = t0 - pd.Timedelta(minutes=1)
        logger.info(
            f"{symbol}: Warm-up complete at {warmup_end}. Daily bars: {len(bars_1d[bars_1d.index<=warmup_end])}. First tradable 1m: {t0}."
        )

        sym_dir = os.path.join(results_dir, symbol)
        utils.ensure_dir(sym_dir)

        pos: risk.Position | None = None
        trades = []
        equity = []
        pnl = 0.0
        consecutive_losses = 0
        cooldown = 0
        daily_start = t0.normalize()
        daily_pnl = 0.0

        for ts in feat.index[feat.index >= t0]:
            row = feat.loc[ts]
            if ts.normalize() > daily_start:
                daily_start = ts.normalize()
                daily_pnl = 0.0

            if pos is None:
                allow_entry = daily_pnl > -cfg["risk"]["daily_loss_limit_pct"] / 100 * max(abs(pnl), 1.0)
                allow_entry &= cooldown == 0
                if allow_entry:
                    prev_close = feat["close"].shift(1).loc[ts]
                    long_trig = (
                        row["macro"] > 0
                        and row["micro"] > cfg["signals"]["micro"].get("min_slope_long", 0.0)
                        and row["compression"]
                        and prev_close <= row["don_high"]
                        and row["close"] > row["don_high"]
                    )
                    short_trig = (
                        row["macro"] < 0
                        and row["micro"] < cfg["signals"]["micro"].get("max_slope_short", 0.0)
                        and row["compression"]
                        and prev_close >= row["don_low"]
                        and row["close"] < row["don_low"]
                    )
                    side = 1 if long_trig else -1 if short_trig else 0
                    if side != 0:
                        # next bar open
                        idx = feat.index.get_loc(ts) + 1
                        if idx < len(feat.index):
                            next_ts = feat.index[idx]
                            next_row = feat.iloc[idx]
                            price = next_row["open"]
                            atr_val = next_row["atr"]
                            sigma_val = next_row["sigma"]
                            notional = portfolio.position_notional(price, sigma_val, cfg["sizing"])
                            qty = notional / price * side
                            fill_price, fee = execution_sim.simulate_fill(
                                price, abs(qty), side, atr_val, cfg["costs"]
                            )
                            sl, tp = risk.initial_sl_tp(fill_price, atr_val, side, cfg["risk"])
                            pos = risk.Position(side=side, qty=qty, entry_price=fill_price, sl=sl, tp=tp)
                            trades.append(
                                {
                                    "symbol": symbol,
                                    "entry_ts": next_ts,
                                    "side": "LONG" if side == 1 else "SHORT",
                                    "qty": qty,
                                    "entry_price": fill_price,
                                    "fee_entry": fee,
                                }
                            )
            else:
                risk.maybe_move_be(pos, row["close"], cfg["risk"])
                risk.maybe_trail(pos, row["atr"], row["close"], cfg["risk"])
                exit_price = risk.check_exit(pos, row)
                if exit_price is not None:
                    fill_price, fee = execution_sim.simulate_fill(
                        exit_price, abs(pos.qty), -pos.side, row["atr"], cfg["costs"]
                    )
                    pnl_trade = (fill_price - pos.entry_price) * pos.qty - fee
                    pnl += pnl_trade
                    daily_pnl += pnl_trade
                    trades[-1].update(
                        {
                            "exit_ts": ts,
                            "exit_price": fill_price,
                            "pnl": pnl_trade,
                            "fee_exit": fee,
                        }
                    )
                    if pnl_trade < 0:
                        consecutive_losses += 1
                        if consecutive_losses >= cfg["risk"].get("cooldown_losses", 3):
                            cooldown = cfg["risk"].get("cooldown_losses", 3)
                            consecutive_losses = 0
                    else:
                        consecutive_losses = 0
                    pos = None
            if cooldown > 0:
                cooldown -= 1
            equity.append({"ts": ts, "equity": pnl})

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(sym_dir, "trades.csv"), index=False)
            trade_returns = trades_df["pnl"] / trades_df["entry_price"].abs()
            all_trade_returns.append(trade_returns)
        else:
            utils.write_advisory(
                os.path.join(sym_dir, "advisory.txt"), f"No trades generated for {symbol}"
            )
        if not equity_df.empty:
            equity_df.to_csv(os.path.join(sym_dir, "equity_curve.csv"), index=False)

    if all_trade_returns:
        returns = pd.concat(all_trade_returns, ignore_index=True)
        summary = metrics.compute_kpis(returns)
        pd.DataFrame([summary]).to_csv(
            os.path.join(results_dir, "summary.csv"), index=False
        )
        metrics_json = metrics.compute_kpis(returns)
        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
            import json

            json.dump(metrics_json, f, indent=2)
        boot = metrics.run_bootstrap_envelopes(returns.values)
        pd.DataFrame(boot).to_csv(
            os.path.join(results_dir, "bootstrap_envelope.csv"), index=False
        )
    utils.save_merged_config(cfg, os.path.join(results_dir, "merged_config.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-horizon backtest")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--results-dir", required=True, help="Where to store artefacts")
    args = parser.parse_args()
    cfg = utils.load_config(args.config)
    run_backtest(cfg, args.results_dir)


if __name__ == "__main__":
    main()

