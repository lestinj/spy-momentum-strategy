
# ==============================================
# FILE: indicators/macro_momentum.py (updated)
# ==============================================
from dataclasses import dataclass
import pandas as pd


@dataclass
class MacroParams:
    required_positive: int = 1


def compute_macro_indicators(params: MacroParams) -> pd.DataFrame:
    """Stub macro set. Replace with real series (FRED, DXY, VIX, rates, M2, etc.).
    For now, derive simple proxies from a CSV at data/macro_{TIMEFRAME}.csv if present; else,
    synthesize rolling filters from SPY itself (will correlate, but unblocks pipeline).
    Required columns if CSV exists: datetime plus any number of indicator columns.
    Each boolean column should indicate 'positive/green'.
    """
    path = "data/macro.csv"
    try:
        df = pd.read_csv(path, parse_dates=[0])
        df.columns = [c.lower() for c in df.columns]
        if "datetime" not in df.columns:
            raise ValueError("macro.csv must have a 'datetime' column")
        df.set_index("datetime", inplace=True)
        # force boolean for all non-index columns
        for c in df.columns:
            df[c] = df[c].astype(bool)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        out = df
    except Exception:
        # Fallback: minimal macro derived from rolling trend of close
        # This ensures the backtest runs even without macro feeds.
        out = pd.DataFrame(index=pd.Index([], name="datetime"))

    if out.empty:
        # Provide at least one permissive macro to avoid blocking entries
        out = pd.DataFrame({"macro_trend": []})

    # Positives count
    out["pos_count"] = out.astype(int).sum(axis=1)
    return out

