import os
import json
import yaml
import pandas as pd

from .data import load_symbol_1m
from .utils import atr as atr_1m_fn
from .cusum import build_cusum_bars
from .adaptive import AdaptiveController
from .waves import WaveGate
from .trigger import Trigger
from .risk import RiskManager
from .regime import TSMOMRegime


def run_for_symbol(cfg: dict, symbol: str, progress_hook=None):
    inputs_dir = cfg['paths']['inputs_dir']
    outputs_dir = cfg['paths']['outputs_dir']
    months = cfg['months']
    os.makedirs(outputs_dir, exist_ok=True)

    df1m = load_symbol_1m(inputs_dir, symbol, months, progress=cfg['logging']['progress'])
    if df1m.empty:
        raise RuntimeError('No 1m data loaded')

    atr1m200 = atr_1m_fn(df1m, int(cfg['waves']['cusum']['atr_window_1m'])).reindex(df1m.index).ffill()

    k = (atr1m200 * float(cfg['waves']['cusum']['k_factor'])).clip(
        lower=atr1m200 * float(cfg['waves']['cusum']['k_min']),
        upper=atr1m200 * float(cfg['waves']['cusum']['k_max'])
    )
    df_event = build_cusum_bars(df1m[['open', 'high', 'low', 'close', 'volume']], k)

    ac = AdaptiveController(cfg, atr1m=atr1m200)

    regime = TSMOMRegime(cfg, df1m)
    waves = WaveGate(cfg, df_event, df1m, ac)
    trigger = Trigger(cfg, df1m, atr1m200, ac)
    risk = RiskManager(cfg, df1m, atr1m200, ac)

    wm = int(cfg['engine']['warmup']['min_1m_bars'])
    wc = int(cfg['engine']['warmup']['min_w2_candidates'])

    i_warm = wm
    ts_warm = df1m.index[min(i_warm, len(df1m) - 1)]
    seen = waves.prewarm_until(ts_warm)
    while seen < wc and i_warm < len(df1m) - 1:
        i_warm = min(len(df1m) - 1, i_warm + wm // 5)
        ts_warm = df1m.index[i_warm]
        seen += waves.prewarm_until(ts_warm)

    start_i = max(wm, i_warm)
    assert ac.ready(start_i, seen), "Warm-start gates not satisfied; increase warmup or data length."

    trades = []
    trade = None
    stride = int(cfg['logging'].get('progress_stride', 200))
    total_bars = len(df1m) - start_i

    iloc = df1m.iloc
    idx = df1m.index

    blockers = {'regime_flat': 0, 'wave_not_armed': 0, 'trigger_fail': 0}

    for i in range(start_i, len(df1m)):
        if progress_hook is not None and ((i - start_i) % stride == 0 or (i + 1) == len(df1m)):
            try:
                progress_hook(symbol, i - start_i + 1, total_bars)
            except Exception:
                pass
        ts = idx[i]

        row = iloc[i]
        if trade is not None and not trade.get('exit'):
            risk.update_trade(trade, row, i)
            risk.check_exit(trade, row)
            if trade.get('exit'):
                r0 = trade['r0']
                if trade['direction'] == 'LONG':
                    r_realized = (trade['exit'] - trade['entry']) / max(1e-9, r0)
                else:
                    r_realized = (trade['entry'] - trade['exit']) / max(1e-9, r0)
                trade['r_realized'] = float(r_realized)
                trades.append(trade)
                trade = None
            continue

        reg = regime.compute_at(ts)
        if reg['dir'] not in ('BULL', 'BEAR'):
            blockers['regime_flat'] += 1
            continue

        wv = waves.compute_at(ts, i)
        if not wv.get('armed', False):
            blockers['wave_not_armed'] += 1
            continue

        tchk = trigger.power_bar_ok(ts, i)
        if not tchk.get('ok', False):
            blockers['trigger_fail'] += 1
            continue

        zret = tchk.get('zret', 0.0)
        if reg['dir'] == 'BULL' and zret < tchk['z_k']:
            blockers['trigger_fail'] += 1
            continue
        if reg['dir'] == 'BEAR' and zret > -tchk['z_k']:
            blockers['trigger_fail'] += 1
            continue

        direction = 'LONG' if reg['dir'] == 'BULL' else 'SHORT'
        entry = row['close']
        stop0 = risk.initial_stop(entry, direction, wv, i)
        r0 = entry - stop0 if direction == 'LONG' else stop0 - entry
        r0 = max(1e-9, r0)

        trade = {
            'symbol': symbol,
            'time': ts.isoformat(),
            'direction': direction,
            'entry': float(entry),
            'stop': float(stop0),
            'r0': float(r0),
            'reason': 'wavegate_momentum',
            'exit': None,
            'exit_reason': None,
            'stop_mode': 'INIT',
            'be_armed': False,
            'tsl_active': False,
            'be_price': float(entry),
            'wave_frame': wv.get('frame', 'event'),
            'adapt_params': {
                'wave': wv.get('params'),
                'trigger': {'zscore_k': tchk['z_k'], 'range_atr_min': tchk['range_atr_min']},
                'risk': risk.thresholds(i),
            },
        }

    if trade is not None and not trade.get('exit'):
        last = iloc[-1]
        trade['exit'] = float(last['close'])
        trade['exit_reason'] = 'EOD'
        if trade['direction'] == 'LONG':
            r_realized = (trade['exit'] - trade['entry']) / max(1e-9, trade['r0'])
        else:
            r_realized = (trade['entry'] - trade['exit']) / max(1e-9, trade['r0'])
        trade['r_realized'] = float(r_realized)
        trades.append(trade)

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(outputs_dir, f"{symbol}_trades.csv"), index=False)

    summary = {'symbol': symbol, 'trades': len(trades_df), 'blockers': blockers}
    if not trades_df.empty:
        summary.update({
            'win_rate': float((trades_df['r_realized'] > 0).mean()),
            'avg_R': float(trades_df['r_realized'].mean()),
            'median_R': float(trades_df['r_realized'].median()),
            'sum_R': float(trades_df['r_realized'].sum()),
            'exits': trades_df['exit_reason'].value_counts().to_dict(),
        })

    with open(os.path.join(outputs_dir, f"{symbol}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    params = {
        'cusum': {
            'k_factor': float(cfg['waves']['cusum']['k_factor']),
            'k_min': float(cfg['waves']['cusum']['k_min']),
            'k_max': float(cfg['waves']['cusum']['k_max']),
            'atr_window_1m': int(cfg['waves']['cusum']['atr_window_1m']),
            'event_bars': int(len(df_event)),
        }
    }
    logs_dir = os.path.join(outputs_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, f"{symbol}_params_run.json"), 'w') as f:
        json.dump(params, f, indent=2)

    return summary


def run_all(config_path: str, progress_hook=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    summaries = []
    for sym in cfg['symbols']:
        try:
            s = run_for_symbol(cfg, sym, progress_hook=progress_hook)
        except Exception as e:
            s = {'symbol': sym, 'error': str(e)}
        summaries.append(s)
    with open(os.path.join(cfg['paths']['outputs_dir'], "combined_summary.json"), 'w') as f:
        json.dump(summaries, f, indent=2)
    return summaries

