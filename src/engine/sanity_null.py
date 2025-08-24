import argparse, os, json, random
import yaml
import numpy as np
import pandas as pd
from .data import load_symbol_1m


def run(cfg, symbol, seed=7, tp_r=1.0, sl_r=1.0, prob_entry=0.002):
    rng = random.Random(seed)
    df = load_symbol_1m(cfg['paths']['inputs_dir'], symbol, cfg['months'], progress=False)
    if df.empty:
        raise RuntimeError("no data")
    out = []
    atr = (df['high']-df['low']).ewm(alpha=1/50, adjust=False).mean()
    for i in range(51, len(df)):
        if rng.random() > prob_entry:
            continue
        row = df.iloc[i]
        entry = float(row['close'])
        a = float(atr.iloc[i])
        r0 = max(1e-9, a*3.0)
        dirn = 1 if rng.random()<0.5 else -1
        sl = entry - dirn*sl_r*r0
        tp = entry + dirn*tp_r*r0
        lo, hi = float(row['low']), float(row['high'])
        exit_price = None
        exit_reason = None
        if dirn==1:
            if lo <= sl:
                exit_price, exit_reason = sl, 'SL'
            if hi >= tp and exit_price is None:
                exit_price, exit_reason = tp, 'TP'
        else:
            if hi >= sl:
                exit_price, exit_reason = sl, 'SL'
            if lo <= tp and exit_price is None:
                exit_price, exit_reason = tp, 'TP'
        if exit_price is None:
            exit_price, exit_reason = float(row['close']), 'EOD'
        r_realized = (exit_price - entry)/r0 if dirn==1 else (entry - exit_price)/r0
        out.append({'time': row.name.isoformat(), 'entry': entry, 'exit': exit_price, 'r': r_realized, 'reason': exit_reason})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    df = run(cfg, args.symbol)
    os.makedirs(cfg['paths']['outputs_dir'], exist_ok=True)
    p = os.path.join(cfg['paths']['outputs_dir'], f"{args.symbol}_null_sanity.csv")
    df.to_csv(p, index=False)
    s = {'n': len(df), 'mean_R': float(df['r'].mean()), 'median_R': float(df['r'].median()), 'pSL': float((df['reason']=='SL').mean()), 'pTP': float((df['reason']=='TP').mean())}
    with open(os.path.join(cfg['paths']['outputs_dir'], f"{args.symbol}_null_sanity.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(s)


if __name__ == "__main__":
    main()
