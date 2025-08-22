
import os, json, yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from .data import load_symbol_1m
from .regime import TSMOMRegime, FLAT
from .waves import WaveGate
from .trigger import momentum_ignition
from .risk import RiskCfg, initial_stop, update_stops, check_exit

from .utils import resample_ohlcv, atr_vec, zscore_logret_vec, body_dom_vec, true_range_vec


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_for_symbol(cfg: dict, symbol: str, progress_hook=None):
    inputs_dir = cfg['paths']['inputs_dir']
    outputs_dir = cfg['paths']['outputs_dir']
    months = cfg['months']
    warmup = int(cfg['warmup']['min_1m_bars'])
    os.makedirs(outputs_dir, exist_ok=True)

    df1m = load_symbol_1m(inputs_dir, symbol, months, progress=cfg['logging']['progress'])
    if len(df1m) < warmup + 500:
        raise RuntimeError("Insufficient 1m bars after loading.")

    df5 = resample_ohlcv(df1m, '5min')
    atr5 = pd.Series(atr_vec(df5['high'].to_numpy(), df5['low'].to_numpy(), df5['close'].to_numpy(),
                              int(cfg['waves']['zigzag']['atr_window'])), index=df5.index)
    regime = TSMOMRegime(cfg, df1m)
    waves = WaveGate(cfg, df5, atr5)

    risk_cfg = RiskCfg(
        atr_window=int(cfg['risk']['atr']['window']),
        sl_mode=cfg['risk']['sl']['mode'],
        sl_atr_mult=float(cfg['risk']['sl']['atr_mult']),
        be_trigger_r=float(cfg['risk']['be']['trigger_r_multiple']),
        tsl_start_r=float(cfg['risk']['tsl']['start_r_multiple']),
        tsl_atr_mult=float(cfg['risk']['tsl']['atr_mult'])
    )

    close = df1m['close'].to_numpy()
    high = df1m['high'].to_numpy()
    low = df1m['low'].to_numpy()
    open_ = df1m['open'].to_numpy()
    prev_close = np.concatenate(([close[0]], close[:-1]))
    atr_arr = atr_vec(high, low, close, risk_cfg.atr_window)
    zret = zscore_logret_vec(close, int(cfg['entry']['momentum']['zscore_window']))
    body_dom_arr = body_dom_vec(open_, high, low, close)
    tr_arr = true_range_vec(high, low, prev_close)
    tr_over_atr = tr_arr / np.maximum(1e-9, atr_arr)
    regime_vec = regime.regime_vec
    idx = df1m.index

    buf = float(cfg['entry']['breakout']['buffer_atr_mult'])
    z_k = float(cfg['entry']['momentum']['zscore_k'])
    min_body = float(cfg['entry']['momentum']['min_body_dom'])
    range_min = float(cfg['entry']['momentum']['range_atr_min'])

    rows = []
    trade = None

    iterator = range(warmup, len(df1m))
    stride = int(cfg['logging'].get('progress_stride', 50))
    total_bars = len(df1m) - warmup

    blockers = {'regime_flat': 0, 'wave_not_armed': 0, 'trigger_fail': 0}
    for i in iterator:
        ts = idx[i]
        price = close[i]
        h = high[i]
        l = low[i]

        if progress_hook is not None and (i == len(df1m) - 1 or ((i - warmup) % max(1, stride) == 0)):
            try:
                progress_hook(symbol, i - warmup, total_bars)
            except Exception:
                pass

        if trade is not None and not trade.get('exit'):
            update_stops(trade, price, atr_arr[i], risk_cfg)
            check_exit(trade, h, l)
            if trade.get('exit'):
                r0 = trade['r0']
                if trade['direction'] == 'LONG':
                    r_realized = (trade['exit'] - trade['entry']) / max(1e-9, r0)
                else:
                    r_realized = (trade['entry'] - trade['exit']) / max(1e-9, r0)
                trade['r_realized'] = float(r_realized)
                rows.append(trade)
                trade = None
            continue

        side = regime_vec[i]
        if side == FLAT:
            blockers['regime_flat'] += 1
            continue

        wave_state = waves.compute_at(ts)
        if (not wave_state.get('armed')) or wave_state.get('dir') != side:
            blockers['wave_not_armed'] += 1
            continue

        trig = momentum_ignition(i, wave_state, side, close, atr_arr, zret, body_dom_arr,
                                 tr_over_atr, buf, z_k, min_body, range_min)
        if not trig:
            blockers['trigger_fail'] += 1
            continue

        entry = price
        stop0 = initial_stop(entry, trig['direction'], wave_state, atr_arr[i], risk_cfg)
        r0 = entry - stop0 if trig['direction'] == 'LONG' else stop0 - entry
        r0 = max(1e-9, r0)
        trade = {
            'symbol': symbol,
            'time': ts.isoformat(),
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
        }

    if trade is not None and not trade.get('exit'):
        last = len(df1m) - 1
        trade['exit'] = float(close[last])
        trade['exit_reason'] = 'EOD'
        if trade['direction'] == 'LONG':
            r_realized = (trade['exit'] - trade['entry']) / max(1e-9, trade['r0'])
        else:
            r_realized = (trade['entry'] - trade['exit']) / max(1e-9, trade['r0'])
        trade['r_realized'] = float(r_realized)
        rows.append(trade)

    trades = pd.DataFrame(rows)
    out_trades = os.path.join(outputs_dir, f"{symbol}_trades.csv")
    trades.to_csv(out_trades, index=False)

    if not trades.empty:
        summary = {
            'symbol': symbol,
            'trades': len(trades),
            'win_rate': float((trades['r_realized'] > 0).mean()),
            'avg_R': float(trades['r_realized'].mean()),
            'median_R': float(trades['r_realized'].median()),
            'sum_R': float(trades['r_realized'].sum()),
            'exits': trades['exit_reason'].value_counts().to_dict(),
            'blockers': blockers
        }
    else:
        summary = {'symbol': symbol, 'trades': 0, 'blockers': blockers}

    with open(os.path.join(outputs_dir, f"{symbol}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

def run_all(config_path: str, progress_hook=None):
    cfg = load_config(config_path)
    summaries = []
    for sym in cfg['symbols']:
        try:
            s = run_for_symbol(cfg, sym, progress_hook=progress_hook)
        except Exception as e:
            s = {'symbol': sym, 'error': str(e)}
        summaries.append(s)
    with open(os.path.join(cfg['paths']['outputs_dir'], "combined_summary.json"), 'w') as f:
        json.dump(summaries, f, indent=2)
    print("Done. Summaries:", summaries)
