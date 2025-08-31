from src.engine.adaptive import AdaptiveController
from src.engine.utils import atr

# Stop-mode ordering and promotion helper
ORDER = {"INIT": 0, "BE": 1, "TSL": 2}

def promote_stop_mode(trade: dict, new_mode: str):
    cur = str(trade.get("stop_mode", "INIT")).upper()
    nm = str(new_mode).upper()
    if ORDER.get(nm, 0) > ORDER.get(cur, 0):
        trade["stop_mode"] = nm


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _exit_fill_price(
    direction: str,
    stop_price: float,
    bar_low: float,
    bar_high: float,
    bar_close: float,
    fill_model: str = "exchange",
    slip_bps: float = 0.0,
) -> float:
    """
    STOP_MARKET fill model on breach.
    - exchange: apply slippage relative to stop, clamp to bar range
    - conservative: LONG -> min(stop, close, low); SHORT -> max(stop, close, high)
    """
    s = float(stop_price); lo = float(bar_low); hi = float(bar_high); c = float(bar_close)
    if fill_model.lower() == "conservative":
        return min(s, c, lo) if direction == "LONG" else max(s, c, hi)
    slip = abs(float(slip_bps)) / 10_000.0
    if direction == "LONG":
        raw = s * (1.0 - slip);  return _clamp(raw, lo, s)
    else:
        raw = s * (1.0 + slip);  return _clamp(raw, s, hi)


class RiskManager:
    def __init__(self, cfg: dict, df1m, atr1m, ac: AdaptiveController):
        self.cfg = cfg
        self.df1m = df1m
        self.atr1m = atr1m
        self.ac = ac

        # Setup ATR risk series
        try:
            if df1m is not None and cfg.get('risk') and cfg['risk'].get('atr'):
                self.atr_risk = atr(df1m, int(cfg['risk']['atr']['window'])).reindex(df1m.index).ffill()
            elif atr1m is not None:
                self.atr_risk = atr1m
            else:
                self.atr_risk = None
        except Exception:
            self.atr_risk = None

        buf = ((cfg.get('risk', {}).get('be', {}) or {}).get('buffer', {}) or {})
        self.be_r_mult = float(buf.get('r_multiple', 0.12))
        self.be_fees = float(buf.get('fees_bps_round_trip', 0.0))
        self.be_slip = float(buf.get('slippage_bps', 0.0))
        self.sl_mode = (cfg.get('risk', {}).get('sl', {}) or {}).get('mode', 'ATR_or_Structure')
        self.sl_atr_mult = float((cfg.get('risk', {}).get('sl', {}) or {}).get('atr_mult', 3.0))

        # Backtest exit model controls
        bex = ((cfg.get('backtest', {}) or {}).get('exits', {}) or {})
        self.bt_fill_model = str(bex.get('fill_model', 'exchange'))
        self.bt_slip_bps = float(bex.get('slip_bps', 0.0))
        self.fees_bps_rt = float(bex.get('fees_bps_round_trip', getattr(self, 'be_fees', 0.0)))

    def thresholds(self, i_bar_1m: int) -> dict:
        return self.ac.risk_params(i_bar_1m)

    def initial_stop(self, entry_price: float, direction: str, wave_state: dict, i_bar: int):
        a = self.atr_risk.iloc[i_bar]
        if direction == 'LONG':
            struct = float(wave_state['w2_low'])
            atr_stop = entry_price - self.sl_atr_mult * a
            return max(struct, atr_stop)
        else:
            struct = float(wave_state['w2_high'])
            atr_stop = entry_price + self.sl_atr_mult * a
            return min(struct, atr_stop)

    def _ensure(self, trade: dict):
        if 'stop_mode' not in trade:
            trade['stop_mode'] = 'INIT'
        if 'be_armed' not in trade:
            trade['be_armed'] = False
        if 'tsl_active' not in trade:
            trade['tsl_active'] = False
        if 'be_price' not in trade:
            trade['be_price'] = trade.get('entry')
        if 'be_floor' not in trade:
            trade['be_floor'] = trade.get('stop')
        if 'stop_trace' not in trade:
            trade['stop_trace'] = []
        if 'last_logged_stop' not in trade:
            trade['last_logged_stop'] = float(trade.get('stop', trade.get('entry', 0.0)))
        if 'last_logged_mode' not in trade:
            trade['last_logged_mode'] = str(trade.get('stop_mode', 'INIT'))

    def update_trade(self, trade: dict, row, i_bar: int):
        if trade.get('exit'):
            return
        self._ensure(trade)

        thr = self.thresholds(i_bar)
        be_trigger_r = thr['be_trigger_r']
        tsl_start_r = thr['tsl_start_r']
        tsl_atr_mult = thr['tsl_atr_mult']

        price = float(row['close'])
        entry = float(trade['entry'])
        r0 = max(1e-9, float(trade['r0']))

        if trade['direction'] == 'LONG':
            r = (price - entry) / r0
            if trade['stop_mode'] == 'INIT' and r >= be_trigger_r:
                buf_r = self.be_r_mult * r0
                friction = entry * (self.be_fees + self.be_slip) / 10000.0
                buffer = max(buf_r, friction)
                stop_price = entry + buffer
                trade['stop'] = max(float(trade['stop']), stop_price)
                trade['be_armed'] = True
                promote_stop_mode(trade, 'BE')
                trade['be_price'] = entry
                trade['be_floor'] = trade['stop']

            if r >= tsl_start_r:
                trailing = price - tsl_atr_mult * float(self.atr_risk.iloc[i_bar])
                new_stop = max(float(trade['stop']), trailing)
                if trade.get('be_armed'):
                    new_stop = max(new_stop, float(trade.get('be_floor', trade['stop'])))
                if new_stop > float(trade['stop']):
                    trade['stop'] = new_stop
                trade['tsl_active'] = True
                promote_stop_mode(trade, 'TSL')
        else:
            r = (entry - price) / r0
            if trade['stop_mode'] == 'INIT' and r >= be_trigger_r:
                buf_r = self.be_r_mult * r0
                friction = entry * (self.be_fees + self.be_slip) / 10000.0
                buffer = max(buf_r, friction)
                stop_price = entry - buffer
                trade['stop'] = min(float(trade['stop']), stop_price)
                trade['be_armed'] = True
                promote_stop_mode(trade, 'BE')
                trade['be_price'] = entry
                trade['be_floor'] = trade['stop']

            if r >= tsl_start_r:
                trailing = price + tsl_atr_mult * float(self.atr_risk.iloc[i_bar])
                new_stop = min(float(trade['stop']), trailing)
                if trade.get('be_armed'):
                    new_stop = min(new_stop, float(trade.get('be_floor', trade['stop'])))
                if new_stop < float(trade['stop']):
                    trade['stop'] = new_stop
                trade['tsl_active'] = True
                promote_stop_mode(trade, 'TSL')

        # Append stop trace on change of stop or mode
        mode_now = str(trade.get('stop_mode', 'INIT'))
        stop_now = float(trade.get('stop', entry))
        changed_stop = abs(stop_now - float(trade.get('last_logged_stop', stop_now))) > 1e-9
        changed_mode = mode_now != trade.get('last_logged_mode')
        if changed_stop or changed_mode:
            ts = getattr(row, 'name', None)
            ts_iso = ts.isoformat() if ts is not None else ''
            trade['stop_trace'].append({
                'time': ts_iso,
                'price': float(price),
                'stop': float(stop_now),
                'mode': mode_now,
                'r': float(r),
                'tsl_active': bool(trade.get('tsl_active', False)),
                'be_armed': bool(trade.get('be_armed', False)),
            })
            trade['last_logged_stop'] = stop_now
            trade['last_logged_mode'] = mode_now

    def check_exit(self, trade: dict, row):
        if trade.get('exit'):
            return
        self._ensure(trade)

        direction = str(trade['direction'])
        stop_px = float(trade['stop'])
        low = float(row['low'])
        high = float(row['high'])
        close = float(row['close'])
        entry = float(trade['entry'])

        breached = (direction == 'LONG' and low <= stop_px) or (direction == 'SHORT' and high >= stop_px)
        if not breached:
            return

        mode = str(trade.get('stop_mode', 'INIT')).upper()
        reason = 'TSL' if mode == 'TSL' else ('BE' if mode == 'BE' else 'SL')

        exit_fill = _exit_fill_price(
            direction=direction,
            stop_price=stop_px,
            bar_low=low,
            bar_high=high,
            bar_close=close,
            fill_model=self.bt_fill_model,
            slip_bps=self.bt_slip_bps,
        )

        acc = (self.cfg.get('risk', {}).get('accounting', {}) or {})
        min_r0_bps = float(acc.get('min_r0_bps', 0.5))
        r0_nominal = float(trade['r0'])
        r0_floor = entry * (min_r0_bps / 10_000.0)
        r0_used = max(1e-9, r0_nominal, r0_floor)

        if direction == 'LONG':
            r_raw = (exit_fill - entry) / r0_used
        else:
            r_raw = (entry - exit_fill) / r0_used
        fee_r = (self.fees_bps_rt / 10_000.0) * entry / r0_used
        r_net = r_raw - fee_r

        trade['exit'] = float(exit_fill)
        trade['exit_reason'] = reason
        trade['exit_r_raw'] = float(r_raw)
        trade['exit_r_fee'] = float(fee_r)
        trade['exit_r'] = float(r_net)


__all__ = [
    'RiskManager',
    '_exit_fill_price',
    '_clamp',
    'promote_stop_mode',
    'ORDER',
]
