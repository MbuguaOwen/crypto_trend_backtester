
from dataclasses import dataclass

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL = "SL","TP","BE","TSL"

@dataclass
class RiskCfg:
    atr_window: int
    sl_mode: str
    sl_atr_mult: float
    be_trigger_r: float
    tsl_start_r: float
    tsl_atr_mult: float

def initial_stop(entry_price: float, direction: str, wave_state: dict, atr_last: float, cfg: RiskCfg):
    a = float(atr_last)
    if direction == "LONG":
        struct = float(wave_state['W2_low'])
        atr_stop = entry_price - cfg.sl_atr_mult * a
        return max(struct, atr_stop)
    else:
        struct = float(wave_state['W2_high'])
        atr_stop = entry_price + cfg.sl_atr_mult * a
        return min(struct, atr_stop)

def _ensure_stop_mode(trade: dict):
    """Populate new stop-mode keys for backward compatibility."""
    if 'stop_mode' not in trade:
        trade['stop_mode'] = "INIT"
    if 'be_armed' not in trade:
        trade['be_armed'] = False
    if 'tsl_active' not in trade:
        trade['tsl_active'] = False
    if 'be_price' not in trade:
        trade['be_price'] = trade.get('entry')


if HAVE_NUMBA:
    @njit
    def _update_stop_kernel(price, entry, stop, r0, be_trigger_r, tsl_start_r, tsl_atr_mult, atr_last, direction, stop_mode, be_armed, tsl_active, be_price):
        r = (price - entry) / r0 if direction == 1 else (entry - price) / r0
        if stop_mode == 0 and r >= be_trigger_r:
            if direction == 1:
                stop = max(stop, entry)
            else:
                stop = min(stop, entry)
            be_armed = True
            stop_mode = 1
            be_price = entry
        if r >= tsl_start_r:
            trailing = price - tsl_atr_mult * atr_last if direction == 1 else price + tsl_atr_mult * atr_last
            if (direction == 1 and trailing > stop) or (direction == -1 and trailing < stop):
                stop = trailing
            tsl_active = True
            stop_mode = 2
        return stop, stop_mode, be_armed, tsl_active, be_price


def update_stops(trade: dict, price: float, atr_last: float, cfg: RiskCfg):
    """Update stop to BE or TSL depending on realized R."""
    if trade.get('exit'):
        return

    _ensure_stop_mode(trade)

    entry = float(trade['entry'])
    r0 = max(1e-9, float(trade['r0']))
    price = float(price)
    if HAVE_NUMBA:
        dir_flag = 1 if trade['direction'] == 'LONG' else -1
        sm_map = {'INIT': 0, 'BE': 1, 'TSL': 2}
        sm_inv = {0: 'INIT', 1: 'BE', 2: 'TSL'}
        stop, sm, be_armed, tsl_active, be_price = _update_stop_kernel(
            price, entry, float(trade['stop']), r0,
            cfg.be_trigger_r, cfg.tsl_start_r, cfg.tsl_atr_mult, float(atr_last),
            dir_flag, sm_map.get(trade['stop_mode'], 0), trade['be_armed'], trade['tsl_active'], trade['be_price']
        )
        trade['stop'] = float(stop)
        trade['be_armed'] = bool(be_armed)
        trade['tsl_active'] = bool(tsl_active)
        trade['be_price'] = float(be_price)
        trade['stop_mode'] = sm_inv[int(sm)]
    else:
        if trade['direction'] == 'LONG':
            r = (price - entry) / r0
            if trade['stop_mode'] == "INIT" and r >= cfg.be_trigger_r:
                trade['stop'] = max(float(trade['stop']), entry)
                trade['be_armed'] = True
                trade['stop_mode'] = "BE"
                trade['be_price'] = entry
            if r >= cfg.tsl_start_r:
                trailing = price - cfg.tsl_atr_mult * float(atr_last)
                new_stop = max(float(trade['stop']), trailing)
                if new_stop > float(trade['stop']):
                    trade['stop'] = new_stop
                trade['tsl_active'] = True
                trade['stop_mode'] = "TSL"
        else:
            r = (entry - price) / r0
            if trade['stop_mode'] == "INIT" and r >= cfg.be_trigger_r:
                trade['stop'] = min(float(trade['stop']), entry)
                trade['be_armed'] = True
                trade['stop_mode'] = "BE"
                trade['be_price'] = entry
            if r >= cfg.tsl_start_r:
                trailing = price + cfg.tsl_atr_mult * float(atr_last)
                new_stop = min(float(trade['stop']), trailing)
                if new_stop < float(trade['stop']):
                    trade['stop'] = new_stop
                trade['tsl_active'] = True
                trade['stop_mode'] = "TSL"


def check_exit(trade: dict, high: float, low: float):
    if trade.get('exit'):
        return

    _ensure_stop_mode(trade)

    if trade['direction'] == 'LONG':
        if low <= float(trade['stop']):
            trade['exit'] = float(trade['stop'])
            trade['exit_reason'] = {
                'INIT': EXIT_SL,
                'BE': EXIT_BE,
                'TSL': EXIT_TSL
            }.get(trade.get('stop_mode', 'INIT'), EXIT_SL)
    else:
        if high >= float(trade['stop']):
            trade['exit'] = float(trade['stop'])
            trade['exit_reason'] = {
                'INIT': EXIT_SL,
                'BE': EXIT_BE,
                'TSL': EXIT_TSL
            }.get(trade.get('stop_mode', 'INIT'), EXIT_SL)