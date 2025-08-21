
import os, json, math, yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from .data import load_symbol_1m
from .regime import TSMOMRegime, FLAT
from .waves import WaveGate
from .trigger import momentum_ignition
from .risk import RiskCfg, initial_stop, update_stops, check_exit

from .utils import atr, resample_ohlcv

def run_for_symbol(cfg: dict, symbol: str, progress_hook=None):
    inputs_dir = cfg['paths']['inputs_dir']
    outputs_dir = cfg['paths']['outputs_dir']
    months = cfg['months']
    warmup = int(cfg['warmup']['min_1m_bars'])
    os.makedirs(outputs_dir, exist_ok=True)

    df1m = load_symbol_1m(inputs_dir, symbol, months, progress=cfg['logging']['progress'])
    if len(df1m) < warmup + 500:
        raise RuntimeError("Insufficient 1m bars after loading.")
    # Precompute resampled frames once (no look-ahead; we slice by ts in-loop)
    df5  = resample_ohlcv(df1m, '5min')
    atr5 = atr(df5, int(cfg['waves']['zigzag']['atr_window']))
    regime = TSMOMRegime(cfg, df1m)
    waves = WaveGate(cfg)

    be_cfg = cfg.get('risk', {}).get('be', {}) or {}
    buf_cfg = be_cfg.get('buffer', {}) or {}
    risk_cfg = RiskCfg(
        atr_window=int(cfg['risk']['atr']['window']),
        sl_mode=cfg['risk']['sl']['mode'],
        sl_atr_mult=float(cfg['risk']['sl']['atr_mult']),
        be_trigger_r=float(be_cfg.get('trigger_r_multiple', 0.0)),
        tsl_start_r=float(cfg['risk']['tsl']['start_r_multiple']),
        tsl_atr_mult=float(cfg['risk']['tsl']['atr_mult']),
        be_buffer_r=float(buf_cfg.get('r_multiple', 0.0)),
        fees_bps_round_trip=float(buf_cfg.get('fees_bps_round_trip', 0.0)),
        slippage_bps=float(buf_cfg.get('slippage_bps', 0.0)),
        be_take_max=True,
    )

    rows = []
    trade = None
    # Precompute ATR for speed
    atr_series = atr(df1m, risk_cfg.atr_window)

    iterator = range(warmup, len(df1m))
    stride = int(cfg['logging'].get('progress_stride', 50))
    total_bars = len(df1m) - warmup

    blockers = {'regime_flat':0, 'wave_not_armed':0, 'trigger_fail':0}
    for i in iterator:
        row = df1m.iloc[i]
        ts = row.name
        # external progress hook (throttled)
        if progress_hook is not None and (i == len(df1m)-1 or ((i - warmup) % max(1, stride) == 0)):
            try:
                progress_hook(symbol, i - warmup, total_bars)
            except Exception:
                pass

        # Manage open trade
        if trade is not None and not trade.get('exit'):
            update_stops(trade, row, atr_series.iloc[i], risk_cfg)
            check_exit(trade, row)
            if trade.get('exit'):
                # R computation
                r0 = trade['r0']
                if trade['direction'] == 'LONG':
                    r_realized = (trade['exit'] - trade['entry'])/max(1e-9, r0)
                else:
                    r_realized = (trade['entry'] - trade['exit'])/max(1e-9, r0)
                trade['r_realized'] = float(r_realized)
                rows.append(trade)
                trade = None
            # continue to next bar even if exit, but allow new entry same bar? We'll skip.
            continue

        # No open trade -> check gates and trigger
        side = regime.decide_at(ts)
        if side == FLAT:
            blockers['regime_flat'] += 1
            continue

        wave_state = waves.compute_at(df5, atr5, ts)
        if (not wave_state.get('armed')) or wave_state.get('dir') != side:
            blockers['wave_not_armed'] += 1
            continue

        # Use df1m up-to-ts for trigger calcs (causal)
        window = df1m.iloc[:i+1]
        trig = momentum_ignition(window, wave_state, side, cfg)
        if not trig:
            blockers['trigger_fail'] += 1
            continue

        # Open trade at close
        entry = row['close']
        stop0 = initial_stop(entry, trig['direction'], wave_state, window, risk_cfg)
        r0 = entry - stop0 if trig['direction']=='LONG' else stop0 - entry
        r0 = max(1e-9, r0)
        trade = {
            'symbol': symbol,
            'time': row.name.isoformat(),
            'direction': trig['direction'],
            'entry': float(entry),
            'stop': float(stop0),
            'r0': float(r0),
            'reason': trig['reason'],
            'exit': None,
            'exit_reason': None,
            # --- diagnostics ---
            'be_armed': False,
            'tsl_active': False,
            'r_peak': 0.0,
            'tsl_lock_R_max': 0.0,
            'be_buffer_R_eff': 0.0,
            'be_buffer_px': 0.0,
            'be_buffer_total_bps': 0.0,
        }

    # if trade still open, close at last
    if trade is not None and not trade.get('exit'):
        last = df1m.iloc[-1]
        trade['exit'] = float(last['close'])
        trade['exit_reason'] = 'EOD'
        if trade['direction'] == 'LONG':
            r_realized = (trade['exit'] - trade['entry'])/max(1e-9, trade['r0'])
        else:
            r_realized = (trade['entry'] - trade['exit'])/max(1e-9, trade['r0'])
        trade['r_realized'] = float(r_realized)
        rows.append(trade)

    # Results
    trades = pd.DataFrame(rows)
    out_trades = os.path.join(outputs_dir, f"{symbol}_trades.csv")
    trades.to_csv(out_trades, index=False)

    summary = {}
    if not trades.empty:
        exit_counts = trades['exit_reason'].value_counts().to_dict()
        sum_by_exit = trades.groupby('exit_reason')['r_realized'].sum().to_dict()
        tsl_series = trades.loc[trades['exit_reason']=='TSL', 'r_realized']
        be_count = int((trades['exit_reason']=='BE').sum())
        sl_count = int((trades['exit_reason']=='SL').sum())
        tsl_count = int((trades['exit_reason']=='TSL').sum())

        tsl_stats = {
            "min": float(tsl_series.min()) if len(tsl_series) else 0.0,
            "p25": float(tsl_series.quantile(0.25)) if len(tsl_series) else 0.0,
            "median": float(tsl_series.median()) if len(tsl_series) else 0.0,
            "p75": float(tsl_series.quantile(0.75)) if len(tsl_series) else 0.0,
            "p90": float(tsl_series.quantile(0.90)) if len(tsl_series) else 0.0,
            "max": float(tsl_series.max()) if len(tsl_series) else 0.0,
            "count": tsl_count,
            "sum_R": float(sum_by_exit.get('TSL', 0.0)),
        }

        sl_sum = float(sum_by_exit.get('SL', 0.0))
        tsl_sum = float(sum_by_exit.get('TSL', 0.0))
        coverage = (tsl_sum / abs(sl_sum)) if sl_sum < 0 else None

        # Optional diagnostics on BE buffers and trailing efficiency
        be_buf_r = trades.loc[trades['be_armed']==True, 'be_buffer_R_eff']
        tsl_eff = trades.loc[trades['exit_reason']=='TSL', ['tsl_lock_R_max','r_peak']]
        tsl_eff['efficiency'] = tsl_eff.apply(lambda x: (x['tsl_lock_R_max'] / x['r_peak']) if x['r_peak']>0 else 0.0, axis=1)

        summary = {
            'symbol': symbol,
            'trades': int(len(trades)),
            'win_rate': float((trades['r_realized']>0).mean()),
            'avg_R': float(trades['r_realized'].mean()),
            'median_R': float(trades['r_realized'].median()),
            'sum_R': float(trades['r_realized'].sum()),
            'exits': exit_counts,
            'sl_count': sl_count,
            'be_count': be_count,
            'tsl_count': tsl_count,
            'tsl_stats': tsl_stats,
            'tsl_coverage_ratio': coverage,
            'be_buffer_R_eff_median': float(be_buf_r.median()) if len(be_buf_r) else 0.0,
            'be_buffer_R_eff_p90': float(be_buf_r.quantile(0.90)) if len(be_buf_r) else 0.0,
            'tsl_efficiency_median': float(tsl_eff['efficiency'].median()) if len(tsl_eff) else 0.0,
            'blockers': blockers
        }
    else:
        summary = {'symbol': symbol, 'trades': 0, 'blockers': blockers}

    with open(os.path.join(outputs_dir, f"{symbol}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # One-line console summary
    try:
        cov_txt = "nan" if summary.get('tsl_coverage_ratio') is None else f"{summary['tsl_coverage_ratio']:.2f}"
        print(
            f"[{symbol}] trades={summary['trades']} | SL={summary.get('sl_count',0)} "
            f"| BE={summary.get('be_count',0)} | TSL={summary.get('tsl_count',0)} "
            f"| sumR={summary['sum_R']:.2f} | TSLcov={cov_txt} | "
            f"TSLmed={summary.get('tsl_stats',{}).get('median',0):.2f}R | "
            f"BEbuf_med={summary.get('be_buffer_R_eff_median',0):.3f}R"
        )
    except Exception:
        pass

    return summary

def run_all(config_path: str, progress_hook=None):
    with open(config_path,'r') as f:
        cfg = yaml.safe_load(f)
    summaries = []
    for sym in cfg['symbols']:
        try:
            s = run_for_symbol(cfg, sym, progress_hook=progress_hook)
        except Exception as e:
            s = {'symbol': sym, 'error': str(e)}
        summaries.append(s)
    # write combined
    with open(os.path.join(cfg['paths']['outputs_dir'], "combined_summary.json"), 'w') as f:
        json.dump(summaries, f, indent=2)
    print("Done. Summaries:", summaries)
