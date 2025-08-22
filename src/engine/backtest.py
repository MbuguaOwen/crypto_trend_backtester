
import os, json, math, yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from .data import load_symbol_1m
from .cusum import build_cusum_bars
from .regime import TSMOMRegime
from .waves import WaveGate
from .trigger import momentum_ignition
from .risk import RiskCfg, initial_stop, update_stops, check_exit

from .utils import atr

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
    # Build CUSUM event-time bars
    cus = cfg['waves']['cusum']
    atr200 = atr(df1m, int(cus['atr_window_1m']))
    k_base = float(cus['k_factor'])
    k_min = float(cus['k_min'])
    k_max = float(cus['k_max'])
    kappa = (atr200 * k_base).clip(lower=atr200 * k_min, upper=atr200 * k_max)
    df_event = build_cusum_bars(df1m[['open','high','low','close','volume']], kappa)

    regime = TSMOMRegime(cfg, df1m)
    waves = WaveGate(cfg, df_event, df1m)

    risk_cfg = RiskCfg(
        atr_window=int(cfg['risk']['atr']['window']),
        sl_mode=cfg['risk']['sl']['mode'],
        sl_atr_mult=float(cfg['risk']['sl']['atr_mult']),
        be_trigger_r=float(cfg['risk']['be']['trigger_r']),
        be_buffer_r_mult=float(cfg['risk']['be']['buffer']['r_multiple']),
        be_fees_bps=float(cfg['risk']['be']['buffer']['fees_bps_round_trip']),
        be_slip_bps=float(cfg['risk']['be']['buffer']['slippage_bps']),
        tsl_start_r=float(cfg['risk']['tsl']['start_r']),
        tsl_atr_mult=float(cfg['risk']['tsl']['atr_mult'])
    )

    rows = []
    trade = None
    # Precompute ATR for speed
    atr_series = atr(df1m, risk_cfg.atr_window)
    # regime tracking statistics
    reg_counts = {"BULL": 0, "BEAR": 0, "FLAT": 0}
    reg_scores = []
    reg_strengths = []

    # log params
    logs_dir = os.path.join(outputs_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, 'params_run.json'), 'w') as f:
        json.dump({
            'cusum_k_factor': k_base,
            'event_bars': int(len(df_event))
        }, f)

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

        # compute regime for this bar (causal)
        reg = regime.compute_at(ts)
        reg_counts[reg['dir']] += 1
        reg_scores.append(reg['score'])
        reg_strengths.append(reg['strength'])

        # Manage open trade first
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
        side = 'LONG' if reg['dir'] == 'BULL' else ('SHORT' if reg['dir'] == 'BEAR' else 'FLAT')
        if side == 'FLAT':
            blockers['regime_flat'] += 1
            continue

        wave_state = waves.compute_at(ts)
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
            'stop_mode': 'INIT',
            'be_armed': False,
            'tsl_active': False,
            'be_price': float(entry),
            'regime_dir': reg['dir'],
            'regime_score': float(reg['score']),
            'regime_strength': float(reg['strength']),
            'wave_frame': wave_state.get('frame', 'event'),
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
        summary = {
            'symbol': symbol,
            'trades': len(trades),
            'win_rate': float((trades['r_realized']>0).mean()),
            'avg_R': float(trades['r_realized'].mean()),
            'median_R': float(trades['r_realized'].median()),
            'sum_R': float(trades['r_realized'].sum()),
            'exits': trades['exit_reason'].value_counts().to_dict(),
            'blockers': blockers,
        }
    else:
        summary = {'symbol': symbol, 'trades': 0, 'blockers': blockers}

    summary.update({
        'regime_bull_bars': reg_counts['BULL'],
        'regime_bear_bars': reg_counts['BEAR'],
        'regime_flat_bars': reg_counts['FLAT'],
        'regime_score_median': float(np.median(reg_scores)) if reg_scores else 0.0,
        'regime_strength_median': float(np.median(reg_strengths)) if reg_strengths else 0.0,
    })

    with open(os.path.join(outputs_dir, f"{symbol}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

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
