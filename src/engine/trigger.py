
import pandas as pd
from .utils import atr, zscore_logret, body_dom, true_range_last

def momentum_ignition(df1m: pd.DataFrame, wave_state: dict, regime_dir: str, cfg: dict):
    if regime_dir not in ("LONG","SHORT"):
        return None
    if not wave_state or not wave_state.get('armed'):
        return None
    if wave_state['dir'] != regime_dir:
        return None

    atr14 = atr(df1m, int(cfg['risk']['atr']['window'])).iloc[-1]
    buf   = float(cfg['entry']['momentum']['buffer_atr_mult'])
    last  = df1m.iloc[-1]
    prev_close = df1m['close'].iloc[-2]

    zret = zscore_logret(df1m['close'], int(cfg['entry']['momentum']['zscore_window'])).iloc[-1]
    body = body_dom(last)
    tratr = true_range_last(last, prev_close) / max(1e-9, atr14)

    z_k = float(cfg['entry']['momentum']['zscore_k'])
    min_body = float(cfg['entry']['momentum']['min_body_dom'])
    range_min = float(cfg['entry']['momentum']['range_atr_min'])

    if regime_dir == 'LONG':
        lvl = float(wave_state['w2_high'] + buf * atr14)
        if (last['close'] >= lvl) and (zret >= z_k) and (body >= min_body) and (tratr >= range_min):
            return {'direction':'LONG', 'level': lvl, 'reason':'wavegate_momentum'}
    else:
        lvl = float(wave_state['w2_low'] - buf * atr14)
        if (last['close'] <= lvl) and (zret <= -z_k) and (body >= min_body) and (tratr >= range_min):
            return {'direction':'SHORT', 'level': lvl, 'reason':'wavegate_momentum'}
    return None
