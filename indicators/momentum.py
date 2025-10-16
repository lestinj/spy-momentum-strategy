from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class MomentumParams:
    fast: int = 20
    slow: int = 50
    rsi_len: int = 10
    rsi_buy: float = 52.0
    rsi_sell: float = 48.0
    z_window: int = 20
    z_entry: float = 0.5
    z_exit: float = 0.0


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    dn = -diff.clip(upper=0)
    rs = up.rolling(n).mean() / (dn.rolling(n).mean() + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n).mean()
    std = s.rolling(n).std(ddof=0)
    return (s - mean) / (std + 1e-12)

def compute_momentum_indicators(price_df: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
    """
    Return EMA fast/slow, RSI, and z-score on returns.
    Robust even if upstream hands us a 1-col DataFrame for 'close'.
    """
    idx = price_df.index

    # Force 'close' to a Series
    close = price_df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").astype(float)

    # Compute indicators
    ema_fast_s = _ema(close, params.fast).reindex(idx)
    ema_slow_s = _ema(close, params.slow).reindex(idx)
    rsi_s      = _rsi(close, params.rsi_len).reindex(idx)
    z_s        = _zscore(close.pct_change().fillna(0), params.z_window).reindex(idx)

    # Assemble (use to_frame so rename logic never hits DataFrame.rename)
    out = pd.concat(
        [
            ema_fast_s.to_frame(name="ema_fast"),
            ema_slow_s.to_frame(name="ema_slow"),
            rsi_s.to_frame(name="rsi"),
            z_s.to_frame(name="zscore"),
        ],
        axis=1,
    )

    return out
