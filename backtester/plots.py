# backtester/plots.py
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def plot_equity_and_drawdown(equity_df: pd.DataFrame, outdir: str):
    os.makedirs(os.path.join(outdir, "charts"), exist_ok=True)
    eq = equity_df.set_index("timestamp")["portfolio_equity"]
    dd = eq / eq.cummax() - 1.0
    plt.figure()
    eq.plot()
    plt.title("Portfolio Equity")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "equity.png"))
    plt.close()
    plt.figure()
    dd.plot()
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "drawdown.png"))
    plt.close()

def plot_bootstrap_envelopes(env: dict, outdir: str):
    os.makedirs(os.path.join(outdir, "charts"), exist_ok=True)
    import numpy as np
    plt.figure()
    x = np.arange(len(env["p50"]))
    plt.plot(x, env["p50"], label="Median")
    plt.plot(x, env["p5"], label="P5")
    plt.plot(x, env["p95"], label="P95")
    plt.title("Bootstrap Equity Envelopes")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "bootstrap_envelopes.png"))
    plt.close()
