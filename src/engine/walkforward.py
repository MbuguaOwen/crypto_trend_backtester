from __future__ import annotations

import os
import json
from copy import deepcopy
from typing import List, Dict

import pandas as pd

from .data import load_symbol_1m
from .backtest import run_for_symbol


def split_months(months: List[str], train: int, test: int, step: int) -> List[Dict[str, List[str]]]:
    folds = []
    n = len(months)
    i = 0
    while i + train < n:
        m_train = months[i : i + train]
        m_test = months[i + train : i + train + test]
        if not m_test:
            break
        folds.append({'train': m_train, 'test': m_test})
        i += step
    return folds


def df_for_months(df_all: pd.DataFrame, months: List[str]) -> pd.DataFrame:
    if not months:
        return df_all.iloc[0:0].copy()
    months_set = set(months)
    mask = df_all.index.strftime('%Y-%m').isin(months_set)
    return df_all.loc[mask].copy()


def run_walkforward(cfg: dict, symbol: str, train: int, test: int, step: int):
    df_all = load_symbol_1m(cfg['paths']['inputs_dir'], symbol, cfg['months'], progress=cfg['logging']['progress'])
    folds = split_months(cfg['months'], train, test, step)
    results = []
    base_out = os.path.join(cfg['paths']['outputs_dir'], 'wf', symbol)
    for k, f in enumerate(folds, start=1):
        m_train, m_test = f['train'], f['test']
        df_train = df_for_months(df_all, m_train)
        df_test = df_for_months(df_all, m_test)
        if df_train.empty or df_test.empty:
            continue
        df_fold = pd.concat([df_train, df_test]).sort_index()
        start_ts = df_test.index[0]
        cfg_fold = deepcopy(cfg)
        outdir = os.path.join(base_out, f'fold_{k:03d}')
        cfg_fold['paths']['outputs_dir'] = outdir
        os.makedirs(outdir, exist_ok=True)
        s = run_for_symbol(cfg_fold, symbol, progress_hook=None, df1m_override=df_fold, trade_start_ts=start_ts)
        base = f"{symbol}_fold_{k:03d}_{m_train[0]}..{m_train[-1]}_{m_test[0]}..{m_test[-1]}"
        tr_path = os.path.join(outdir, f"{symbol}_trades.csv")
        if os.path.exists(tr_path):
            os.rename(tr_path, os.path.join(outdir, f"{base}_trades.csv"))
        sum_path = os.path.join(outdir, f"{symbol}_summary.json")
        if os.path.exists(sum_path):
            os.rename(sum_path, os.path.join(outdir, f"{base}_summary.json"))
        s.update({
            'fold_id': k,
            'train_months': m_train,
            'test_months': m_test,
            'trade_start_ts': start_ts.isoformat(),
            'bars_train': int(len(df_train)),
            'bars_test': int(len(df_test)),
        })
        results.append(s)
        print(f"FOLD {k:03d} [train: {m_train[0]}..{m_train[-1]} | test: {m_test[0]}] → trades={s.get('trades',0)}, sum_R={s.get('sum_R',0):.2f}, win_rate={s.get('win_rate',0):.2%}")
    agg_keys = ['sum_r_sl', 'sum_r_be', 'sum_r_tsl', 'sum_r_sl_overshoot', 'sum_r_realized', 'SL_count', 'BE_count', 'TSL_count', 'trades']
    aggregate = {k: 0.0 for k in agg_keys}
    aggregate['SL_count'] = 0
    aggregate['BE_count'] = 0
    aggregate['TSL_count'] = 0
    aggregate['trades'] = 0
    win_acc = 0.0
    avg_acc = 0.0
    for r in results:
        for k in agg_keys:
            if k in r:
                aggregate[k] += r.get(k, 0.0)
        win_acc += r.get('win_rate', 0.0) * r.get('trades', 0)
        avg_acc += r.get('avg_R', 0.0) * r.get('trades', 0)
    tot = aggregate.get('trades', 0)
    if tot > 0:
        aggregate['win_rate'] = win_acc / tot
        aggregate['avg_R'] = avg_acc / tot
    else:
        aggregate['win_rate'] = 0.0
        aggregate['avg_R'] = 0.0
    with open(os.path.join(base_out, 'aggregate.json'), 'w') as f:
        json.dump({'folds': results, 'aggregate': aggregate}, f, indent=2)
    return results
