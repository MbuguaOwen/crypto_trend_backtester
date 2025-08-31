from .adaptive import AdaptiveController
from .utils import atr


# --- NEW: small helper to clamp a value between lo and hi
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --- NEW: compute the actual exit fill price given a stop breach and model
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
    Models STOP_MARKET fills on breach.
    - exchange: apply slippage to stop and clamp to bar range (realistic)
    - conservative: LONG uses min(stop, close, low); SHORT uses max(stop, close, high)
    """
    s = float(stop_price)
    lo = float(bar_low)
    hi = float(bar_high)
    c = float(bar_close)

    if fill_model.lower() == "conservative":
        if direction == "LONG":
            return min(s, c, lo)
        else:
            return max(s, c, hi)

    # exchange-like model
    slip = abs(float(slip_bps)) / 10_000.0
    if direction == "LONG":
        # stop triggers a SELL → price no better than stop; slippage moves fill below stop
        raw = s * (1.0 - slip)
        return _clamp(raw, lo, s)
    else:
        # stop triggers a BUY to cover short → price no better than stop; slippage above stop
        raw = s * (1.0 + slip)
        return _clamp(raw, s, hi)


class RiskManager:
    def __init__(self, cfg: dict, df1m, atr1m, ac: AdaptiveController):
        self.cfg = cfg
        self.df1m = df1m
        self.atr1m = atr1m
        self.ac = ac

        # safe ATR risk series initialization (allow tests/minimal cfg)
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

        # ---- NEW: backtest exit model controls ----
        bex = ((cfg.get('backtest', {}) or {}).get('exits', {}) or {})
        self.bt_fill_model = str(bex.get('fill_model', 'exchange'))
        self.bt_slip_bps = float(bex.get('slip_bps', 0.0))

        # fees for realized-R; prefer exits.fees_bps_round_trip if present,
        # else reuse whatever you already use for BE fees/buffer (self.be_fees)
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
        # stop trace init
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
                trade['stop_mode'] = 'BE'
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
                trade['stop_mode'] = 'TSL'
        else:
            r = (entry - price) / r0
            if trade['stop_mode'] == 'INIT' and r >= be_trigger_r:
                buf_r = self.be_r_mult * r0
                friction = entry * (self.be_fees + self.be_slip) / 10000.0
                buffer = max(buf_r, friction)
                stop_price = entry - buffer
                trade['stop'] = min(float(trade['stop']), stop_price)
                trade['be_armed'] = True
                trade['stop_mode'] = 'BE'
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
                trade['stop_mode'] = 'TSL'

        # ---- Append stop trace on change of stop or mode ----
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
        # already exited?
        if trade.get('exit'):
            return
        self._ensure(trade)

        direction = str(trade['direction'])
        stop_px = float(trade['stop'])
        low = float(row['low'])
        high = float(row['high'])
        close = float(row['close'])
        entry = float(trade['entry'])
        # --- Floor r0 to avoid absurd R due to tiny denominators
        acc = (self.cfg.get('risk', {}).get('accounting', {}) or {})
        min_r0_bps = float(acc.get('min_r0_bps', 0.5))
        r0 = max(1e-9, float(trade['r0']))
        r0_floor = entry * (min_r0_bps / 10_000.0)
        r0_used = max(r0, r0_floor)

        breached = (direction == 'LONG' and low <= stop_px) or (direction == 'SHORT' and high >= stop_px)
        if not breached:
            return

        # classify reason using your current stop_mode semantics (unchanged)
        mode = str(trade.get('stop_mode', 'INIT')).upper()
        if mode == 'TSL':
            reason = 'TSL'
        elif mode == 'BE':
            reason = 'BE'
        else:
            reason = 'SL'

        # --- NEW: compute realistic exit fill
        exit_fill = _exit_fill_price(
            direction=direction,
            stop_price=stop_px,
            bar_low=low,
            bar_high=high,
            bar_close=close,
            fill_model=self.bt_fill_model,
            slip_bps=self.bt_slip_bps,
        )

        # --- NEW: realized R including fees (round-trip) in R units
        # raw R from price move (use r0_used)
        if direction == 'LONG':
            r_raw = (exit_fill - entry) / r0_used
        else:
            r_raw = (entry - exit_fill) / r0_used

        # fee impact in R units (approx: bps of notional divided by risk distance)
        fee_r = (self.fees_bps_rt / 10_000.0) * entry / r0_used
        r_net = r_raw - fee_r

        # record exit + diagnostics
        trade['exit'] = float(exit_fill)
        trade['exit_reason'] = reason
        trade['r0_used'] = float(r0_used)
        trade['exit_r_raw'] = float(r_raw)
        trade['exit_r_fee'] = float(fee_r)
        trade['exit_r'] = float(r_net)

        # Guard: exit_r must be finite to avoid corrupt KPIs
        import math
        assert math.isfinite(trade['exit_r']), 'exit_r not finite; check r0, fee math'

