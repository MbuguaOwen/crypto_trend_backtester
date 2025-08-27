import pandas as pd

from .data import load_symbol_1m
from .backtest import run_for_symbol


def df_for_months(df_all: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    if not months:
        return df_all.iloc[0:0]
    mask = pd.Series(False, index=df_all.index)
    for m in months:
        y, mo = m.split('-')
        mask |= ((df_all.index.year == int(y)) & (df_all.index.month == int(mo)))
    return df_all[mask]


def split_months(months, train, test, step):
    out = []
    i = 0
    while True:
        tr = months[i:i+train]
        te = months[i+train:i+train+test]
        if not tr or not te:
            break
        out.append({'train': tr, 'test': te})
        i += max(1, step)
    return out


def run_walkforward(cfg: dict, symbol: str, train: int, test: int, step: int, progress_hook=None):
    df_all = load_symbol_1m(cfg['paths']['inputs_dir'], symbol, cfg['months'],
                            progress=cfg['logging']['progress'])
    folds = split_months(cfg['months'], train, test, step)
    results = []
    for k, f in enumerate(folds, start=1):
        df_train = df_for_months(df_all, f['train'])
        df_test = df_for_months(df_all, f['test'])
        if df_train.empty or df_test.empty:
            continue
        df_fold = pd.concat([df_train, df_test]).sort_index()
        start_ts = df_test.index[0]

        def hook(sym, done, total, fid=k):
            if progress_hook is not None:
                progress_hook(f"{sym}:F{fid}", done, total)

        s = run_for_symbol(cfg, symbol,
                           progress_hook=hook,
                           df1m_override=df_fold,
                           trade_start_ts=start_ts)
        s['fold_id'] = k
        s['train_months'] = f['train']
        s['test_months'] = f['test']
        results.append(s)
    return results

