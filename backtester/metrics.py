# backtester/metrics.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Any, List
from scipy.stats import skew, kurtosis, norm

def compute_kpis(returns: pd.Series) -> Dict[str,Any]:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r)==0:
        return {"n":0,"cagr":0,"stdev":0,"sharpe":0,"sortino":0,"max_dd":0,"calmar":0,"profit_factor":1.0,"expectancy":0,"hit_rate":0,"skew":0,"kurtosis":0}
    n = len(r)
    mean = r.mean()
    stdev = r.std(ddof=0)
    downside = r[r<0].std(ddof=0)
    sharpe = (mean / stdev) * np.sqrt(365*24*60) if stdev>0 else 0.0
    sortino = (mean / downside) * np.sqrt(365*24*60) if (downside is not None and downside>0) else 0.0
    eq = (1+r).cumprod()
    peak = eq.cummax()
    dd = (eq/peak - 1).min()
    max_dd = abs(dd)
    cagr = eq.iloc[-1]**( (365*24*60)/n ) - 1 if n>0 else 0.0
    profit_factor = (r[r>0].sum() / abs(r[r<0].sum())) if abs(r[r<0].sum())>0 else np.inf
    expectancy = r.mean()
    hit_rate = (r>0).mean()
    sk = float(skew(r, bias=False)) if n>2 else 0.0
    kt = float(kurtosis(r, fisher=True, bias=False)) if n>3 else 0.0
    calmar = (cagr / max_dd) if max_dd>0 else np.inf
    return {"n": n, "cagr": float(cagr), "stdev": float(stdev), "sharpe": float(sharpe), "sortino": float(sortino),
            "max_dd": float(max_dd), "calmar": float(calmar), "profit_factor": float(profit_factor),
            "expectancy": float(expectancy), "hit_rate": float(hit_rate), "skew": sk, "kurtosis": kt}

def probabilistic_sharpe_ratio(sr_hat: float, n: int, sr_bench: float = 0.0, skew: float = 0.0, kurt: float = 0.0) -> float:
    if n <= 1:
        return 0.5
    z = (sr_hat - sr_bench) * np.sqrt((n - 1) / (1 - skew*sr_hat + ((kurt - 1)/4.0)*sr_hat**2))
    from scipy.stats import norm
    return float(norm.cdf(z))

def deflated_sharpe_ratio(sr_hat: float, n_trials: int, skew: float, kurt: float, n: int) -> float:
    if n<=1:
        return 0.0
    import numpy as np
    sr_expected_max = np.sqrt(2*np.log(n_trials)) - (np.log(np.pi) + np.log(np.log(n_trials))) / (2*np.sqrt(2*np.log(n_trials))) if n_trials>1 else 0.0
    sr_var = (1 + (skew*sr_hat)/2 - (kurt - 3)/4 * sr_hat**2) / (n - 1)
    z = (sr_hat - sr_expected_max) / np.sqrt(max(sr_var, 1e-12))
    from scipy.stats import norm
    return float(norm.cdf(z))

def _stationary_bootstrap_idx(n: int, p: float, rng):
    idx = []
    i = rng.integers(0, n)
    while len(idx) < n:
        idx.append(i)
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i = (i + 1) % n
    return np.array(idx[:n])

def run_bootstrap_envelopes(returns: np.ndarray, reps: int = 1000, method: str = "stationary", block_p: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(returns)
    curves = []
    for _ in range(reps):
        if method == "stationary":
            idx = _stationary_bootstrap_idx(n, block_p, rng)
        else:
            idx = rng.integers(0, n, size=n)
        r = returns[idx]
        eq = (1+r).cumprod()
        curves.append(eq)
    curves = np.vstack(curves)
    pct5 = np.percentile(curves, 5, axis=0)
    pct50 = np.percentile(curves, 50, axis=0)
    pct95 = np.percentile(curves, 95, axis=0)
    return {"p5": pct5, "p50": pct50, "p95": pct95}

def white_reality_check(strategy_returns: List[np.ndarray], reps: int = 1000, block_p: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    def sharpe(x):
        m = np.mean(x)
        s = np.std(x, ddof=0)
        return (m/s)*np.sqrt(365*24*60) if s>0 else 0.0
    sr_obs = max(sharpe(np.asarray(r)) for r in strategy_returns)
    n = len(strategy_returns[0])
    centered = [np.asarray(r) - np.mean(r) for r in strategy_returns]
    srs = []
    for _ in range(reps):
        max_sr = -1e9
        for r in centered:
            idx = _stationary_bootstrap_idx(n, block_p, rng)
            rb = r[idx]
            max_sr = max(max_sr, sharpe(rb))
        srs.append(max_sr)
    pval = float((np.sum(np.array(srs) >= sr_obs) + 1) / (reps + 1))
    return {"pvalue": pval, "sr_obs": float(sr_obs)}
