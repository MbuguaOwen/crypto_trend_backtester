
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
    be_buffer_r_mult: float
    be_fees_bps: float
    be_slip_bps: float
    tsl_start_r: float
    tsl_atr_mult: float

def initial_stop(entry_price: float, direction: str, wave_state: dict, df1m: pd.DataFrame, cfg: RiskCfg):
    a = atr(df1m, cfg.atr_window).iloc[-1]
    if direction == "LONG":
        struct = float(wave_state['w2_low'])
        atr_stop = entry_price - cfg.sl_atr_mult * a
        return max(struct, atr_stop)
    else:
        struct = float(wave_state['w2_high'])
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
    if 'be_floor' not in trade:
        trade['be_floor'] = trade.get('stop')


def update_stops(trade: dict, row, atr_last, cfg: RiskCfg):
    """Update stop to BE or TSL depending on realized R."""
    if trade.get('exit'):
        return

    _ensure_stop_mode(trade)

    price = float(row['close'])
    entry = float(trade['entry'])
    r0 = max(1e-9, float(trade['r0']))

    if trade['direction'] == 'LONG':
        r = (price - entry) / r0

        # Arm BE once
        if trade['stop_mode'] == "INIT" and r >= cfg.be_trigger_r:
            buf_r = cfg.be_buffer_r_mult * r0
            friction = entry * (cfg.be_fees_bps + cfg.be_slip_bps) / 10000.0
            buffer = max(buf_r, friction)
            stop_price = entry + buffer
            trade['stop'] = max(float(trade['stop']), stop_price)
            trade['be_armed'] = True
            trade['stop_mode'] = "BE"
            trade['be_price'] = entry
            trade['be_floor'] = trade['stop']

        # Activate / maintain TSL
        if r >= cfg.tsl_start_r:
            trailing = price - cfg.tsl_atr_mult * float(atr_last)
            new_stop = max(float(trade['stop']), trailing)
            if trade.get('be_armed'):
                new_stop = max(new_stop, float(trade.get('be_floor', trade['stop'])))
            if new_stop > float(trade['stop']):
                trade['stop'] = new_stop
            trade['tsl_active'] = True
            trade['stop_mode'] = "TSL"

    else:  # SHORT
        r = (entry - price) / r0

        # Arm BE once
        if trade['stop_mode'] == "INIT" and r >= cfg.be_trigger_r:
            buf_r = cfg.be_buffer_r_mult * r0
            friction = entry * (cfg.be_fees_bps + cfg.be_slip_bps) / 10000.0
            buffer = max(buf_r, friction)
            stop_price = entry - buffer
            trade['stop'] = min(float(trade['stop']), stop_price)
            trade['be_armed'] = True
            trade['stop_mode'] = "BE"
            trade['be_price'] = entry
            trade['be_floor'] = trade['stop']

        # Activate / maintain TSL
        if r >= cfg.tsl_start_r:
            trailing = price + cfg.tsl_atr_mult * float(atr_last)
            new_stop = min(float(trade['stop']), trailing)
            if trade.get('be_armed'):
                new_stop = min(new_stop, float(trade.get('be_floor', trade['stop'])))
            if new_stop < float(trade['stop']):
                trade['stop'] = new_stop
            trade['tsl_active'] = True
            trade['stop_mode'] = "TSL"


def check_exit(trade: dict, row):
    if trade.get('exit'):
        return

    _ensure_stop_mode(trade)

    if trade['direction'] == 'LONG':
        if row['low'] <= float(trade['stop']):
            trade['exit'] = float(trade['stop'])
            trade['exit_reason'] = {
                'INIT': EXIT_SL,
                'BE': EXIT_BE,
                'TSL': EXIT_TSL
            }.get(trade.get('stop_mode', 'INIT'), EXIT_SL)
    else:
        if row['high'] >= float(trade['stop']):
            trade['exit'] = float(trade['stop'])
            trade['exit_reason'] = {
                'INIT': EXIT_SL,
                'BE': EXIT_BE,
                'TSL': EXIT_TSL
            }.get(trade.get('stop_mode', 'INIT'), EXIT_SL)
