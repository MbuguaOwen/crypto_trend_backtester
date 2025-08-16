from typing import Dict
import pandas as pd


class TSMOMRegime:
    """Minimal regime classifier based on simple price momentum."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def classify(self, tf_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        directions = {}
        for tf, df in tf_data.items():
            if df.shape[0] < 2:
                continue
            directions[tf] = "long" if df["close"].iloc[-1] >= df["close"].iloc[-2] else "short"
        # majority vote
        longs = list(directions.values()).count("long")
        shorts = list(directions.values()).count("short")
        regime = "long" if longs >= shorts else "short"
        return {"direction": regime}
