
# ==============================================
# FILE: indicators/alignment_checker.py (updated)
# ==============================================
from dataclasses import dataclass
import pandas as pd


@dataclass
class AlignmentParams:
    min_aligned: int = 2
    lookback: int = 10


def compute_alignment(df: pd.DataFrame, params: AlignmentParams) -> pd.DataFrame:
    """Compute a simple breadth/alignment measure using momentum as proxy.
    If you have sector/industry ETFs, replace logic to count how many are in uptrend.
    Here we create an aligned_count from rolling occurrences where ema_fast > ema_slow.
    """
    aligned_flag = (df["ema_fast"] > df["ema_slow"]).astype(int)
    aligned_count = aligned_flag.rolling(params.lookback).sum().fillna(0)
    return pd.DataFrame({"aligned_count": aligned_count}, index=df.index)
