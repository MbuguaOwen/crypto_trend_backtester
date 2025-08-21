
from dataclasses import dataclass
import pandas as pd
from .utils import atr

# --- keep existing constants ---
EXIT_SL, EXIT_TP, EXIT_BE, EXIT_TSL = "SL","TP","BE","TSL"


@dataclass
class RiskCfg:
    atr_window: int
    sl_mode: str
    sl_atr_mult: float
    be_trigger_r: float
    tsl_start_r: float
    tsl_atr_mult: float
    be_buffer_r: float = 0.0
    fees_bps_round_trip: float = 0.0
    slippage_bps: float = 0.0
    be_take_max: bool = True

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

EPS = 1e-9


def _locked_r(trade, stop_price: float) -> float:
    """Return locked R for current stop relative to entry."""
    r0 = max(EPS, float(trade['r0']))
    if trade['direction'] == 'LONG':
        return (float(stop_price) - float(trade['entry'])) / r0
    else:
        return (float(trade['entry']) - float(stop_price)) / r0


def _be_buffer_components(trade: dict, cfg) -> dict:
    """
    Compute BE buffer components in both price and R units, then pick the effective (max).
    cfg.be_buffer_r: optional fixed R multiple (from YAML buffer.r_multiple)
    cfg.fees_bps_round_trip, cfg.slippage_bps: bps to convert to price
    """
    entry = float(trade['entry'])
    r0 = max(EPS, float(trade['r0']))

    # Components
    r_buf_r = float(getattr(cfg, 'be_buffer_r', 0.0))
    bps_fees = float(getattr(cfg, 'fees_bps_round_trip', 0.0))
    bps_slip = float(getattr(cfg, 'slippage_bps', 0.0))
    total_bps = max(0.0, bps_fees + bps_slip)

    # Convert bps to price
    px_buf_bps = entry * (total_bps / 10000.0)
    # Convert bps to R
    r_buf_bps_as_R = px_buf_bps / r0

    # Effective R buffer (max of components)
    r_eff = max(0.0, r_buf_r, r_buf_bps_as_R)
    # Effective price buffer
    px_eff = r_eff * r0

    return {
        "r_buf_r": r_buf_r,
        "r_buf_bps_as_R": r_buf_bps_as_R,
        "r_eff": r_eff,
        "px_eff": px_eff,
        "total_bps": total_bps
    }


def update_stops(trade: dict, row, atr_last, cfg: RiskCfg):
    """
    Update BE and trailing stops; enforce monotonic stop; track r_peak and TSL lock.
    Assumes cfg has:
      - be_trigger_r (float)
      - tsl_start_r (float)
      - tsl_atr_mult (float)
      - be_buffer_r (float), fees_bps_round_trip (float), slippage_bps (float)
    """
    if trade.get('exit'):
        return

    price = float(row['close'])
    r0 = max(EPS, float(trade['r0']))
    long = (trade['direction'] == 'LONG')

    # Unrealized R now & peak
    if long:
        r_now = (price - trade['entry']) / r0
    else:
        r_now = (trade['entry'] - price) / r0
    trade['r_peak'] = max(float(trade.get('r_peak', 0.0)), float(r_now))

    # --- (1) Arm BE with buffer when threshold reached ---
    if not trade.get('be_armed', False) and r_now >= float(cfg.be_trigger_r):
        # compute effective BE buffer
        bebuf = _be_buffer_components(trade, cfg)
        trade['be_buffer_R_eff'] = bebuf['r_eff']
        trade['be_buffer_px'] = bebuf['px_eff']
        trade['be_buffer_total_bps'] = bebuf['total_bps']

        if long:
            # move stop to entry + buffer
            candidate = float(trade['entry']) + bebuf['px_eff']
            trade['stop'] = max(float(trade['stop']), candidate)
        else:
            candidate = float(trade['entry']) - bebuf['px_eff']
            trade['stop'] = min(float(trade['stop']), candidate)

        trade['be_floor'] = candidate  # price level at/just beyond entry covering costs
        trade['be_armed'] = True

    # Determine BE floor (after BE armed); before that, floor=entry
    if trade.get('be_armed', False):
        be_floor = float(trade.get('be_floor', trade['entry']))
    else:
        be_floor = float(trade['entry'])

    # --- (2) Trailing stop after tsl_start_r; respect BE floor/ceiling ---
    if r_now >= float(cfg.tsl_start_r):
        trade['tsl_active'] = True
        if long:
            candidate = price - float(cfg.tsl_atr_mult) * float(atr_last)
            # Guard: once BE armed, never let trailing dip below BE floor
            candidate = max(candidate, be_floor + EPS) if trade.get('be_armed', False) else candidate
            # Monotonic stop
            trade['stop'] = max(float(trade['stop']), candidate)
        else:
            candidate = price + float(cfg.tsl_atr_mult) * float(atr_last)
            candidate = min(candidate, be_floor - EPS) if trade.get('be_armed', False) else candidate
            trade['stop'] = min(float(trade['stop']), candidate)

        # Track best locked R achieved by the trailer
        trade['tsl_lock_R_max'] = max(
            float(trade.get('tsl_lock_R_max', 0.0)),
            float(_locked_r(trade, float(trade['stop'])))
        )


def check_exit(trade: dict, row):
    """
    Exit precedence at stop-touch:
      - If BE armed and stop <= BE floor (+EPS for longs; -EPS for shorts) → BE
      - Else if TSL active and stop beyond BE floor → TSL
      - Else → SL
    """
    if trade.get('exit'):
        return

    long = (trade['direction'] == 'LONG')
    be_floor = float(trade.get('be_floor', trade['entry']))

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
