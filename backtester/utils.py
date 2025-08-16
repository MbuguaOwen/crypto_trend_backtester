# backtester/utils.py
"""Utility helpers for the event driven backtester.

This module intentionally keeps dependencies light and provides a couple of
helpers that are reused across the new multi‑horizon backtest stack:

* deterministic seeding
* configuration loading/merging
* filesystem helpers
* structured logging setup

The functions are deliberately small so they can be easily unit tested.  The
logger returned by :func:`get_logger` writes to the results directory and logs
to stdout which keeps the CLI ergonomic while still persisting artefacts.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import yaml


def seed_everything(seed: int = 42) -> None:
    """Seed Python's ``random`` and ``numpy`` RNGs."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """Create ``path`` if it does not already exist."""

    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return the result."""

    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def save_merged_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist the final resolved configuration for exact reproducibility."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_logger(name: str, log_path: str) -> logging.Logger:
    """Return a structured logger writing to ``log_path`` and stdout."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def write_advisory(path: str, msg: str) -> None:
    """Write a short advisory message when no trades were produced."""

    with open(path, "w", encoding="utf-8") as f:
        f.write(msg.strip() + "\n")

