
def momentum_ignition(i: int, wave_state: dict, regime_dir: str, close, atr_arr, zret, body_dom_arr, tr_over_atr,
                      buf: float, z_k: float, min_body: float, range_min: float):
    if regime_dir not in ("LONG", "SHORT"):
        return None
    if not wave_state or not wave_state.get('armed'):
        return None
    if wave_state.get('dir') != regime_dir:
        return None

    atr14 = float(atr_arr[i])
    price = float(close[i])

    if regime_dir == 'LONG':
        lvl = float(wave_state['W2_high'] + buf * atr14)
        if (price >= lvl) and (zret[i] >= z_k) and (body_dom_arr[i] >= min_body) and (tr_over_atr[i] >= range_min):
            return {'direction': 'LONG', 'level': lvl, 'reason': 'wavegate_momentum'}
    else:
        lvl = float(wave_state['W2_low'] - buf * atr14)
        if (price <= lvl) and (zret[i] <= -z_k) and (body_dom_arr[i] >= min_body) and (tr_over_atr[i] >= range_min):
            return {'direction': 'SHORT', 'level': lvl, 'reason': 'wavegate_momentum'}
    return None
