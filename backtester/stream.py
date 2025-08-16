from __future__ import annotations

import heapq
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Aliases for auto-detecting CSV columns
_ALIASES = {
    "timestamp": ["timestamp", "time", "t", "T"],
    "price":     ["price", "p"],
    "quantity":  ["quantity", "qty", "q", "amount", "volume"],
}

def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse ms-epoch or ISO8601 strings to UTC Timestamps, robustly."""
    if np.issubdtype(series.dtype, np.number):
        return pd.to_datetime(series, unit="ms", utc=True)

    # Try strict ISO8601 first, then general
    out = pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")
    if out.isna().any():
        out = pd.to_datetime(series, utc=True, errors="coerce")
    if out.isna().any():
        bad = series[out.isna()].iloc[:5].tolist()
        raise ValueError(f"Unparseable timestamps detected (first 5): {bad}")
    return out

def _detect_tick_columns(path: str) -> Dict[str, str]:
    """Read header only and map standard names -> actual CSV column names."""
    hdr = pd.read_csv(path, nrows=0)
    cols = [c.strip() for c in hdr.columns.tolist()]
    lower = {c.lower(): c for c in cols}
    out: Dict[str, str] = {}
    for std, choices in _ALIASES.items():
        found = None
        for cand in choices:
            if cand.lower() in lower:
                found = lower[cand.lower()]
                break
        if not found:
            raise ValueError(f"Required column '{std}' not found. Available: {cols}")
        out[std] = found
    return out

class TickCSVStream:
    """
    Chunked tick reader:
      - Auto-detects timestamp/price/quantity columns
      - Parses ms-epoch or ISO8601
      - Filters by [start_ts, end_ts)
      - Uses C engine for chunked streaming (pyarrow doesn't support chunksize)
    """
    def __init__(
        self,
        path: str,
        symbol: str,
        chunksize: int = 1_000_000,
        start_ts: pd.Timestamp | None = None,
        end_ts: pd.Timestamp | None = None,
    ):
        self.path = path
        self.symbol = symbol
        self.chunksize = chunksize
        self.start_ts = start_ts
        self.end_ts = end_ts

        self._chunk_iter = None
        self._buf = None
        self._i = 0

        self._colmap = _detect_tick_columns(self.path)
        self._usecols = [self._colmap["timestamp"], self._colmap["price"], self._colmap["quantity"]]

    def _ensure_iter(self):
        if self._chunk_iter is None:
            # IMPORTANT: pyarrow engine does NOT support chunksize in pandas;
            # always use the C engine for streaming.
            self._chunk_iter = pd.read_csv(
                self.path,
                usecols=self._usecols,
                chunksize=self.chunksize,
                engine="c",
            )

    def _load_chunk(self) -> bool:
        self._ensure_iter()
        try:
            chunk = next(self._chunk_iter)
        except StopIteration:
            self._buf = None
            self._i = 0
            return False

        # Standardize column names
        chunk = chunk.rename(
            columns={
                self._colmap["timestamp"]: "timestamp",
                self._colmap["price"]: "price",
                self._colmap["quantity"]: "quantity",
            }
        )

        # Parse + filter
        chunk["timestamp"] = _parse_timestamps(chunk["timestamp"])
        if self.start_ts is not None:
            chunk = chunk[chunk["timestamp"] >= self.start_ts]
        if self.end_ts is not None:
            chunk = chunk[chunk["timestamp"] < self.end_ts]

        if chunk.empty:
            # Nothing in this chunk; load the next one
            return self._load_chunk()

        # Clean numeric
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce")
        chunk = (
            chunk.dropna(subset=["timestamp", "price", "quantity"])
                 .sort_values("timestamp")
                 .reset_index(drop=True)
        )

        if chunk.empty:
            return self._load_chunk()

        self._buf = chunk
        self._i = 0
        return True

    def next_row(self) -> Optional[Dict[str, Any]]:
        if self._buf is None or self._i >= len(self._buf):
            if not self._load_chunk():
                return None
        row = self._buf.iloc[self._i]
        self._i += 1
        return {
            "timestamp": row["timestamp"],
            "symbol": self.symbol,
            "price": float(row["price"]),
            "quantity": float(row["quantity"]),
        }

class MultiSymbolMerger:
    """K-way merge on timestamp from multiple TickCSVStreams."""
    def __init__(self, streams: List[TickCSVStream]):
        self.streams = streams
        self.heap: List[Tuple[pd.Timestamp, int, Dict[str, Any]]] = []

    def __iter__(self):
        for idx, s in enumerate(self.streams):
            row = s.next_row()
            if row is not None:
                self.heap.append((row["timestamp"], idx, row))
        heapq.heapify(self.heap)
        return self

    def __next__(self) -> Dict[str, Any]:
        if not self.heap:
            raise StopIteration
        _, idx, row = heapq.heappop(self.heap)
        nxt = self.streams[idx].next_row()
        if nxt is not None:
            heapq.heappush(self.heap, (nxt["timestamp"], idx, nxt))
        return row
