from __future__ import annotations
import math
import numpy as np, pandas as pd
from typing import Optional, Dict, Any
from .bocpd import BOCPD

# ------------ small helpers ------------
def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    alpha = 1.0 / max(1, n)
    return tr.ewm(alpha=alpha, adjust=False).mean()

def _logret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1)).fillna(0.0)

def _rv_percentile(logr: pd.Series, rv_lookback: int, pct_window: int) -> float:
    rv = logr.rolling(rv_lookback, min_periods=rv_lookback).std(ddof=0)
    if len(rv) < pct_window + 5 or rv.iloc[-1] != rv.iloc[-1]:
        return 1.0
    recent = float(rv.iloc[-1])
    hist = rv.iloc[-pct_window-1:-1].values
    return float((hist <= recent).sum()) / max(1, len(hist))  # 0..1

def _assert_no_legacy(cfg: dict):
    # Raise if legacy trigger settings exist
    legacy_paths = [
        ("strategy","trigger","gating"),
        ("strategy","trigger","compression"),
        ("strategy","trigger","breakout","primary"),
        ("strategy","trigger","breakout","ksigma"),
        ("entry","thrust"),
        ("entry","retest"),
    ]
    for p in legacy_paths:
        d = cfg
        ok = True
        for k in p:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                ok = False; break
        if ok:
            raise ValueError(f"Legacy config detected at {'.'.join(p)} — remove it for BOCPD-only mode.")

class BreakoutAfterCompression:
    """
    BOCPD-only trigger:
    - Require regime ∈ {LONG, SHORT}
    - Require realized-vol squeeze (percentile)
    - Require price pierce prior-bar Donchian ± ATR buffer
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        _assert_no_legacy(cfg)
        mode = str(cfg.get("entry", {}).get("mode", "")).lower()
        if mode and mode != "bocpd_squeeze_breakout":
            raise ValueError("Only entry.mode 'bocpd_squeeze_breakout' is supported in this build.")
        self.state: Dict[str, Any] = {}  # per-symbol: BOCPD + last_fire_bar

    def _S(self, sym: str) -> Dict[str, Any]:
        ent = self.cfg.get("entry", {}).get("bocpd", {})
        if sym not in self.state:
            self.state[sym] = {
                "bocpd": BOCPD(
                    hazard_lambda=float(ent.get("hazard_lambda", 200)),
                    rmax=int(ent.get("rmax", 600)),
                    mu0=float(ent.get("prior", {}).get("mu0", 0.0)),
                    kappa0=float(ent.get("prior", {}).get("kappa0", 1e-3)),
                    alpha0=float(ent.get("prior", {}).get("alpha0", 1.0)),
                    beta0=float(ent.get("prior", {}).get("beta0", 1.0)),
                ),
                "last_fire_bar": -10**9
            }
        return self.state[sym]

    def check(self, df_1m: pd.DataFrame, df_hist: pd.DataFrame, regime_dir: str) -> Optional[Dict[str, Any]]:
        # Gate regime + history length
        if regime_dir not in ("LONG", "SHORT"):
            return None
        if len(df_hist) < 200:
            return None

        sym = "SYMBOL"
        if "symbol" in df_hist.columns:
            try: sym = str(df_hist["symbol"].iloc[-1])
            except Exception: pass

        ent = (self.cfg.get("entry", {}) or {}).get("bocpd", {}) or {}
        min_cp = float(ent.get("min_cp_prob", 0.80))
        cooldown = int(ent.get("cooldown_bars", 30))
        fresh = int(ent.get("freshness_bars", 120))
        sqz = ent.get("squeeze", {}) or {}
        rv_lb = int(sqz.get("rv_lookback", 30))
        pctw = int(sqz.get("pct_window", 300))
        max_pct = float(sqz.get("max_pct", 0.35))

        brk = self.cfg.get("trigger", {}).get("breakout", {}) or {}
        don_len = int(brk.get("donchian_lookback", 25))
        buf_mult = float(brk.get("buffer_atr_mult", 0.25))

        atr_win = int(self.cfg.get("risk", {}).get("atr", {}).get("window", 14))
        atr_val = float(_atr(df_hist[["high","low","close"]], n=atr_win).iloc[-1])

        prior = df_hist.iloc[:-1]
        if len(prior) < max(2, don_len):
            return None
        don_high = float(prior["high"].tail(don_len).max())
        don_low  = float(prior["low"].tail(don_len).min())
        base_long = don_high + buf_mult * atr_val
        base_short= don_low  - buf_mult * atr_val

        S = self._S(sym)
        bars_since = len(df_hist) - S["last_fire_bar"]
        if bars_since <= cooldown:
            return None

        # BOCPD update on latest log return
        lr = _logret(df_hist["close"])
        cp_prob = float(S["bocpd"].update(float(lr.iloc[-1])))

        # Squeeze gate via rv percentile
        squeeze_pct = _rv_percentile(lr, rv_lb, pctw)
        if not (cp_prob >= min_cp and squeeze_pct <= max_pct):
            return None

        last_close = float(df_hist["close"].iloc[-1])
        if regime_dir == "LONG" and last_close >= base_long:
            S["last_fire_bar"] = len(df_hist)
            return {"direction":"LONG","level": float(base_long),
                    "reason":"bocpd_squeeze_breakout",
                    "meta":{"cp_prob": round(cp_prob,4), "squeeze_pct": round(squeeze_pct,3), "don_level": float(base_long)}}
        if regime_dir == "SHORT" and last_close <= base_short:
            S["last_fire_bar"] = len(df_hist)
            return {"direction":"SHORT","level": float(base_short),
                    "reason":"bocpd_squeeze_breakout",
                    "meta":{"cp_prob": round(cp_prob,4), "squeeze_pct": round(squeeze_pct,3), "don_level": float(base_short)}}
        return None
