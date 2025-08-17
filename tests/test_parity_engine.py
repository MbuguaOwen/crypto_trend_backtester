import os, pandas as pd, yaml, sys, pytest
sys.path.append(os.getcwd())
from backtests.parity_backtest import warmup_bars_required, run_symbol

def test_warmup():
    with open('configs/default.yaml','r') as f:
        cfg = yaml.safe_load(f)
    w = warmup_bars_required(cfg)
    assert isinstance(w, int) and w > 0

def test_run_symbol(tmp_path):
    import shutil
    with open('configs/default.yaml','r') as f:
        cfg = yaml.safe_load(f)
    out = tmp_path / 'out.csv'
    try:
        df = run_symbol(cfg, 'BTCUSDT', str(out))
    except FileNotFoundError:
        pytest.skip('data files missing')
    assert out.exists()
    if len(df) > 0:
        assert set(df['exit_type']).issubset({'SL','TP','BE','TSL','MACRO_FLIP'})
