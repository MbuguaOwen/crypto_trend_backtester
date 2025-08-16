# backtester/utils.py
from __future__ import annotations
import os, json, random
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_merged_config(cfg, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def write_advisory(path: str, msg: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")
