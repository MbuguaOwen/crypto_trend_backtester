# backtester/sensitivity.py
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def tornado_from_grid(grid_df: pd.DataFrame, outdir: str, metric: str = "calmar"):
    os.makedirs(os.path.join(outdir, "charts"), exist_ok=True)
    params_expanded = grid_df["params"].apply(pd.Series)
    df = pd.concat([params_expanded, grid_df.drop(columns=["params"])], axis=1)
    med = df[metric].median()
    ranges = []
    for col in params_expanded.columns:
        grp = df.groupby(col)[metric].agg(["min","max"])
        lo = (grp["min"] - med).abs().max()
        hi = (grp["max"] - med).abs().max()
        ranges.append((col, float(lo), float(hi)))
    ranges.sort(key=lambda x: x[1]+x[2], reverse=True)
    labels = [r[0] for r in ranges]
    lows = [r[1] for r in ranges]
    highs = [r[2] for r in ranges]
    y = np.arange(len(labels))
    plt.figure()
    plt.barh(y, highs, left=0)
    plt.barh(y, -np.array(lows), left=0)
    plt.yticks(y, labels)
    plt.title(f"Tornado Plot on {metric}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "charts", "tornado.png"))
    plt.close()
