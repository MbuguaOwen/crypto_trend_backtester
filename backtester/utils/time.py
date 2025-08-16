
import pandas as pd

MS_IN_MIN = 60_000

def annualize_vol(vol_per_min: float) -> float:
    return vol_per_min * (1440 * 365) ** 0.5
