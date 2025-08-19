
from dataclasses import dataclass
import pandas as pd
from .utils import atr

EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL = "SL","TP","BE","TSL"

@dataclass
class RiskCfg:
    atr_window: int
    sl_mode: str
    sl_atr_mult: float
    be_trigger_r: float
    tsl_start_r: float
    tsl_atr_mult: float

def initial_stop(entry_price: float, direction: str, wave_state: dict, df1m: pd.DataFrame, cfg: RiskCfg):
    a = atr(df1m, cfg.atr_window).iloc[-1]
    if direction == "LONG":
        struct = float(wave_state['W2_low'])
        atr_stop = entry_price - cfg.sl_atr_mult * a
        return max(struct, atr_stop)
    else:
        struct = float(wave_state['W2_high'])
        atr_stop = entry_price + cfg.sl_atr_mult * a
        return min(struct, atr_stop)

def update_stops(trade: dict, row, atr_last, cfg: RiskCfg):
    """
    trade: dict with entry, stop, direction, r0 (initial risk distance)
    row: last 1m row
    """
    if trade.get('exit'):
        return
    price = row['close']
    if trade['direction'] == 'LONG':
        up = price - trade['entry']
        r = up / max(1e-9, trade['r0'])
        # BE
        if (not trade.get('be_armed', False)) and r >= cfg.be_trigger_r:
            trade['stop'] = max(trade['stop'], trade['entry'])
            trade['be_armed'] = True
        # TSL
        if r >= cfg.tsl_start_r:
            trailing = price - cfg.tsl_atr_mult * atr_last
            trade['stop'] = max(trade['stop'], trailing)
    else:
        down = trade['entry'] - price
        r = down / max(1e-9, trade['r0'])
        if (not trade.get('be_armed', False)) and r >= cfg.be_trigger_r:
            trade['stop'] = min(trade['stop'], trade['entry'])
            trade['be_armed'] = True
        if r >= cfg.tsl_start_r:
            trailing = price + cfg.tsl_atr_mult * atr_last
            trade['stop'] = min(trade['stop'], trailing)

def check_exit(trade: dict, row):
    if trade.get('exit'):
        return
    if trade['direction'] == 'LONG':
        if row['low'] <= trade['stop']:
            trade['exit'] = trade['stop']; trade['exit_reason'] = EXIT_SL if not trade.get('be_armed') else EXIT_TSL
        # TP optional (not used by default)
    else:
        if row['high'] >= trade['stop']:
            trade['exit'] = trade['stop']; trade['exit_reason'] = EXIT_SL if not trade.get('be_armed') else EXIT_TSL
