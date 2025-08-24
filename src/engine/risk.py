from .adaptive import AdaptiveController
from .utils import atr


class RiskManager:
    def __init__(self, cfg: dict, df1m, atr1m, ac: AdaptiveController):
        self.cfg = cfg
        self.df1m = df1m
        self.atr1m = atr1m
        self.ac = ac

        self.atr_risk = atr(df1m, int(cfg['risk']['atr']['window'])).reindex(df1m.index).ffill()

        buf = cfg['risk']['be']['buffer']
        self.be_r_mult = float(buf['r_multiple'])
        self.be_fees = float(buf['fees_bps_round_trip'])
        self.be_slip = float(buf['slippage_bps'])
        self.sl_mode = cfg['risk']['sl']['mode']
        self.sl_atr_mult = float(cfg['risk']['sl']['atr_mult'])

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
        if trade.get('exit'):
            return
        self._ensure(trade)
        if trade['direction'] == 'LONG':
            if row['low'] <= float(trade['stop']):
                trade['exit'] = float(trade['stop'])
                trade['exit_reason'] = {
                    'INIT': 'SL',
                    'BE': 'BE',
                    'TSL': 'TSL',
                }.get(trade.get('stop_mode', 'INIT'), 'SL')
        else:
            if row['high'] >= float(trade['stop']):
                trade['exit'] = float(trade['stop'])
                trade['exit_reason'] = {
                    'INIT': 'SL',
                    'BE': 'BE',
                    'TSL': 'TSL',
                }.get(trade.get('stop_mode', 'INIT'), 'SL')

