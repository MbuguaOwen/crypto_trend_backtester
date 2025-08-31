import os
import json
import yaml
import numpy as np
import pandas as pd

from .utils import atr as atr_1m_fn
from .cusum import build_cusum_bars
from .adaptive import AdaptiveController
from .waves import WaveGate
from .trigger import Trigger
from risk import RiskManager
from .regime import TSMOMRegime


# --- R accounting helper (pure, side-effect free) ---
def _apply_r_accounting(trade: dict, cfg: dict) -> dict:
    """
    Enforce:
      - r_sl == -1.0 for SL exits (if sl_exact_neg1), else <= 0
      - r_be >= 0 only for BE
      - r_tsl >= 0 only for TSL
      - r_realized_adjusted == r_sl + r_be + r_tsl (+ r_sl_overshoot)
      - floor r0 to avoid absurd R due to tiny denominators
    """
    acc = (cfg.get('risk', {}).get('accounting', {}) or {})
    sl_exact_neg1 = bool(acc.get('sl_exact_neg1', True))
    record_overs = bool(acc.get('record_sl_overshoot', True))
    min_r0_bps = float(acc.get('min_r0_bps', 0.5))

    # Inputs (must exist in trade dict)
    direction = str(trade['direction']).upper()
    entry = float(trade['entry'])
    exitp = float(trade['exit'])
    r0_orig = float(trade.get('r0', 0.0))
    exit_reason = str(trade.get('exit_reason', 'SL')).upper()

    # Floor r0 in bps of entry; store r0_used
    r0_floor = max(1e-9, entry * (min_r0_bps / 1e4))
    r0_used = max(r0_floor, r0_orig)
    trade['r0_used'] = float(r0_used)

    # Recompute realized R with r0_used
    if direction == 'LONG':
        r_realized = (exitp - entry) / r0_used
    else:
        r_realized = (entry - exitp) / r0_used
    trade['r_realized'] = float(r_realized)

    # Buckets
    r_sl = 0.0
    r_be = 0.0
    r_tsl = 0.0
    r_sl_overshoot = 0.0

    if exit_reason == 'SL':
        if sl_exact_neg1:
            r_sl = -1.0
            if record_overs:
                # Overshoot (+/-) relative to the ideal -1R; audit only
                r_sl_overshoot = float(r_realized - r_sl)
        else:
            # If not forcing -1R, still ensure SL can't be positive
            r_sl = float(min(0.0, r_realized))
    elif exit_reason == 'BE':
        r_be = float(max(0.0, r_realized))
    elif exit_reason == 'TSL':
        r_tsl = float(max(0.0, r_realized))
    else:
        # EOD or other bookkeeping exits: leave buckets zero; r_realized remains for net
        pass

    trade['r_sl'] = float(r_sl)
    trade['r_be'] = float(r_be)
    trade['r_tsl'] = float(r_tsl)
    trade['r_sl_overshoot'] = float(r_sl_overshoot)

    # Reconciliation column
    trade['r_realized_adjusted'] = float(r_sl + r_be + r_tsl + r_sl_overshoot)

    return trade


def run_for_symbol(cfg: dict, symbol: str, progress_hook=None,
                   df1m_override=None, trade_start_ts=None):
    inputs_dir = cfg['paths']['inputs_dir']
    outputs_dir = cfg['paths']['outputs_dir']
    months = cfg['months']
    os.makedirs(outputs_dir, exist_ok=True)

    if df1m_override is not None:
        df1m = df1m_override
    else:
        from .data import load_symbol_1m
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

    if trade_start_ts is not None:
        ts_boundary = pd.to_datetime(trade_start_ts, utc=True)
        seen += waves.prewarm_until(ts_boundary)
        start_i = max(wm, df1m.index.get_indexer([ts_boundary], method='pad')[0])
    else:
        start_i = max(wm, i_warm)

    assert ac.ready(start_i, seen), "Warm-start gates not satisfied; increase warmup or data length."

    trades = []
    trade_id = 0
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
                trade = _apply_r_accounting(trade, cfg)
                logs_dir = os.path.join(outputs_dir, 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                st = trade.get('stop_trace', [])
                if st:
                    import pandas as _pd
                    _pd.DataFrame(st).to_csv(
                        os.path.join(logs_dir, f"{symbol}_trade_{trade.get('id','NA')}_stoptrace.csv"),
                        index=False,
                    )
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
            'id': int(trade_id),
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
        trade_id += 1
    if trade is not None and not trade.get('exit'):
        last = iloc[-1]
        trade['exit'] = float(last['close'])
        trade['exit_reason'] = 'EOD'
        trade = _apply_r_accounting(trade, cfg)
        logs_dir = os.path.join(outputs_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        st = trade.get('stop_trace', [])
        if st:
            import pandas as _pd
            _pd.DataFrame(st).to_csv(
                os.path.join(logs_dir, f"{symbol}_trade_{trade.get('id','NA')}_stoptrace.csv"),
                index=False,
            )
        trades.append(trade)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        for col in ('r_sl', 'r_be', 'r_tsl', 'r_sl_overshoot', 'r_realized_adjusted', 'r0_used'):
            if col not in trades_df.columns:
                trades_df[col] = 0.0

        # --- Sanity warnings (non-fatal) ---
        try:
            acc = (cfg.get('risk', {}).get('accounting', {}) or {})
            sl_exact_neg1 = bool(acc.get('sl_exact_neg1', True))

            sl_cnt = int((trades_df['exit_reason'].str.upper() == 'SL').sum())
            sl_sum = float(trades_df['r_sl'].sum())
            be_neg = int((trades_df['r_be'] < -1e-9).sum())
            tsl_neg = int((trades_df['r_tsl'] < -1e-9).sum())
            recon = float((trades_df['r_realized_adjusted'] - trades_df['r_realized']).abs().sum())

            if sl_exact_neg1 and abs(sl_sum + sl_cnt) > 1e-6:
                print(f"[WARN] sum(r_sl) {sl_sum:.6f} != -SL_count {-sl_cnt}")
            if be_neg:
                print(f"[WARN] {be_neg} BE rows < 0 (expected >= 0)")
            if tsl_neg:
                print(f"[WARN] {tsl_neg} TSL rows < 0 (expected >= 0)")
            if recon > 1e-6:
                print(f"[WARN] Buckets != r_realized by total {recon:.6f}")
        except Exception:
            pass

    trades_df.to_csv(os.path.join(outputs_dir, f"{symbol}_trades.csv"), index=False)

    summary = {'symbol': symbol, 'trades': len(trades_df), 'blockers': blockers}
    if not trades_df.empty:
        import numpy as np

        # Prefer exit_r (post-fix). Fallback to r_realized_adjusted if any non-finite values.
        r_pref = trades_df['exit_r'].copy() if 'exit_r' in trades_df.columns else None
        if r_pref is None or (~np.isfinite(r_pref)).any():
            r_pref = trades_df['r_realized_adjusted'].copy()

        r_pref = r_pref.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        summary['trades'] = int(len(trades_df))
        summary['win_rate'] = float((r_pref > 0).mean()) if len(trades_df) else 0.0
        summary['avg_R'] = float(r_pref.mean()) if len(trades_df) else 0.0
        summary['median_R'] = float(r_pref.median()) if len(trades_df) else 0.0
        summary['sum_R'] = float(r_pref.sum()) if len(trades_df) else 0.0
        summary['exits'] = trades_df['exit_reason'].value_counts().to_dict()

        # Sanity guards
        assert (~np.isfinite(r_pref)).sum() == 0, 'Non-finite realized R in KPIs'

        # Optional reconciliation
        if {'r_realized_adjusted', 'r_realized'}.issubset(trades_df.columns):
            diff_total = float((trades_df['r_realized_adjusted'] - trades_df['r_realized']).abs().sum())
            if diff_total > 1e-6:
                print(f"[WARN] bucket vs realized diff total: {diff_total:.6f}")

        # BE realism check: 75th percentile of BE exits should be < +0.25R
        if {'exit_reason', 'exit_r'}.issubset(trades_df.columns):
            be = trades_df.query("exit_reason == 'BE'")
            if len(be) > 0:
                q75 = float(be['exit_r'].quantile(0.75))
                assert q75 < 0.25, f"BE exits unrealistically positive (q75={q75:.2f}); likely TSL misclassification"

        # keep existing aggregates
        summary.update({
            'sum_r_sl': float(trades_df['r_sl'].sum()),
            'sum_r_be': float(trades_df['r_be'].sum()),
            'sum_r_tsl': float(trades_df['r_tsl'].sum()),
            'sum_r_sl_overshoot': float(trades_df.get('r_sl_overshoot', 0.0).sum()),
            'sum_r_realized': float(trades_df['r_realized'].sum()),
            'sum_r_realized_adjusted': float(trades_df.get('r_realized_adjusted', trades_df['r_realized']).sum()),
            'SL_count': int((trades_df['exit_reason'].str.upper() == 'SL').sum()),
            'BE_count': int((trades_df['exit_reason'].str.upper() == 'BE').sum()),
            'TSL_count': int((trades_df['exit_reason'].str.upper() == 'TSL').sum()),
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

    # Blocker telemetry for quick diagnosis
    if 'blockers' in summary:
        b = summary['blockers']
        try:
            print(f"[BLOCKERS] regime_flat={b.get('regime_flat',0)} wave_not_armed={b.get('wave_not_armed',0)} trigger_fail={b.get('trigger_fail',0)}")
        except Exception:
            pass

    return summary


def run_all(config_path: str, progress_hook=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    summaries = []
    agg_keys = [
        'sum_r_sl',
        'sum_r_be',
        'sum_r_tsl',
        'sum_r_sl_overshoot',
        'sum_r_realized',
        'sum_r_realized_adjusted',
        'SL_count',
        'BE_count',
        'TSL_count',
        'sum_R',
        'trades',
    ]
    aggregate = {k: 0.0 for k in agg_keys}
    aggregate['SL_count'] = 0
    aggregate['BE_count'] = 0
    aggregate['TSL_count'] = 0
    aggregate['trades'] = 0
    for sym in cfg['symbols']:
        try:
            s = run_for_symbol(cfg, sym, progress_hook=progress_hook)
        except Exception as e:
            s = {'symbol': sym, 'error': str(e)}
        summaries.append(s)
        for k in agg_keys:
            if k in s:
                aggregate[k] += s.get(k, 0.0)
    with open(os.path.join(cfg['paths']['outputs_dir'], "combined_summary.json"), 'w') as f:
        json.dump({'summaries': summaries, 'aggregate': aggregate}, f, indent=2)
    return summaries

