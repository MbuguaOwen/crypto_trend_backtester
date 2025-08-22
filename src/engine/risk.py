
from dataclasses import dataclass
import pandas as pd
from .utils import atr

EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL = "SL","TP","BE","TSL"
EPS = 1e-9


def _locked_r(trade, stop_price: float) -> float:
    r0 = max(EPS, float(trade['r0']))
    if trade['direction'] == 'LONG':
        return (float(stop_price) - float(trade['entry'])) / r0
    else:
        return (float(trade['entry']) - float(stop_price)) / r0

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

def update_stops(trade: dict, row, atr_last: float, cfg: dict):
    """
    BE = exactly entry (no buffer). TSL respects BE floor and never widens.
    Required cfg keys:
      - be_trigger_r = cfg['risk']['be']['trigger_r_multiple']
      - tsl_start_r  = cfg['risk']['tsl']['start_r_multiple']
      - tsl_atr_mult = cfg['risk']['tsl']['atr_mult']
    """
    if trade.get('exit'):
        return

    price = float(row['close'])
    entry = float(trade['entry'])
    r0 = max(EPS, float(trade['r0']))
    long = (trade['direction'] == 'LONG')

    # unrealized R now, update r_peak
    r_now = (price - entry)/r0 if long else (entry - price)/r0
    trade['r_peak'] = max(float(trade.get('r_peak', 0.0)), float(r_now))

    be_trigger_r = float(cfg['risk']['be']['trigger_r_multiple'])
    tsl_start_r  = float(cfg['risk']['tsl']['start_r_multiple'])
    tsl_atr_mult = float(cfg['risk']['tsl']['atr_mult'])

    # 1) Arm BE (no buffer) → stop = entry
    if not trade.get('be_armed', False) and r_now >= be_trigger_r:
        if long:
            trade['stop'] = max(float(trade['stop']), entry)
        else:
            trade['stop'] = min(float(trade['stop']), entry)
        trade['be_floor'] = entry  # used as floor/ceiling for TSL classification
        trade['be_armed'] = True

    be_floor = float(trade.get('be_floor', entry))

    # 2) Trailing stop after tsl_start_r; respect BE floor; never widen
    if r_now >= tsl_start_r:
        trade['tsl_active'] = True
        if long:
            candidate = price - tsl_atr_mult * float(atr_last)
            # if BE armed, TSL must be strictly past entry
            if trade.get('be_armed', False):
                candidate = max(candidate, be_floor + EPS)
            trade['stop'] = max(float(trade['stop']), candidate)  # monotonic
        else:
            candidate = price + tsl_atr_mult * float(atr_last)
            if trade.get('be_armed', False):
                candidate = min(candidate, be_floor - EPS)
            trade['stop'] = min(float(trade['stop']), candidate)

        # Track max locked R by the trailer
        trade['tsl_lock_R_max'] = max(
            float(trade.get('tsl_lock_R_max', 0.0)),
            float(_locked_r(trade, float(trade['stop'])))
        )


def check_exit(trade: dict, row):
    """
    Stop-touch classification with precedence:
      - if BE armed and stop <= entry (+EPS long / -EPS short) → BE
      - elif TSL active and stop beyond entry → TSL
      - else → SL
    """
    if trade.get('exit'):
        return

    entry = float(trade['entry'])
    be_floor = float(trade.get('be_floor', entry))
    long = (trade['direction'] == 'LONG')

    if long:
        if float(row['low']) <= float(trade['stop']):
            trade['exit'] = float(trade['stop'])
            if trade.get('be_armed', False):
                if trade['stop'] <= be_floor + EPS:
                    trade['exit_reason'] = EXIT_BE
                elif trade.get('tsl_active', False) and trade['stop'] > be_floor + EPS:
                    trade['exit_reason'] = EXIT_TSL
                else:
                    trade['exit_reason'] = EXIT_BE
            else:
                trade['exit_reason'] = EXIT_SL
    else:
        if float(row['high']) >= float(trade['stop']):
            trade['exit'] = float(trade['stop'])
            if trade.get('be_armed', False):
                if trade['stop'] >= be_floor - EPS:
                    trade['exit_reason'] = EXIT_BE
                elif trade.get('tsl_active', False) and trade['stop'] < be_floor - EPS:
                    trade['exit_reason'] = EXIT_TSL
                else:
                    trade['exit_reason'] = EXIT_BE
            else:
                trade['exit_reason'] = EXIT_SL
