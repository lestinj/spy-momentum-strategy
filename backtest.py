# ==============================================
# FILE: backtest.py (updated with yfinance loader)
# ==============================================
import os
import sys
from datetime import datetime

import pandas as pd

# --- Local imports -----------------------------------------------------------
from indicators.momentum import compute_momentum_indicators, MomentumParams
from indicators.macro_momentum import compute_macro_indicators, MacroParams
from indicators.alignment_checker import compute_alignment, AlignmentParams

# --- External data feed ------------------------------------------------------
# Price data via yfinance; install with: pip install yfinance
try:
    import yfinance as yf
except Exception as e:
    yf = None

# --- Tunable knobs to target ~2 trades/week ----------------------------------
MOM_PARAMS = MomentumParams(
    fast=20,           # was 50
    slow=50,           # was 200
    rsi_len=10,        # was 14
    rsi_buy=52.0,      # was 55
    rsi_sell=48.0,     # was 45
    z_window=20,
    z_entry=0.5,       # was 1.0
    z_exit=0.0         # was 0.5
)

MACRO_PARAMS = MacroParams(
    required_positive=1   # relax: only 1 macro green required
)

ALIGN_PARAMS = AlignmentParams(
    min_aligned=2,    # relax: was 4
    lookback=10
)

SYMBOL = "SPY"
TIMEFRAME = "1h"       # try "30m" if you still want more trades
START = "2024-06-01"   # widen if you want more sample
END = None             # None -> now

# Map pandas/yfinance-friendly intervals
_YF_INTERVALS = {
    "1h": "60m",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1d": "1d",
}

# --- Data loading helpers ----------------------------------------------------

def load_price_data(symbol: str = SYMBOL, timeframe: str = TIMEFRAME, start: str = START, end: str | None = END) -> pd.DataFrame:
    """Load price data from yfinance with DatetimeIndex and [open, high, low, close, volume].
    yfinance returns tz-aware UTC; we make it tz-naive for safe joins.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    interval = _YF_INTERVALS.get(timeframe, timeframe)
    df = yf.download(tickers=symbol, interval=interval, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}. Try a different START or TIMEFRAME (e.g., '30m').")

    # Standardize columns and index
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "close",
        "Volume": "volume",
    })[["open", "high", "low", "close", "volume"]]

    # yfinance index is tz-aware UTC; normalize to tz-naive
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)

    return df.sort_index()


def load_macro_data() -> pd.DataFrame:
    """Load/compute macro signals via indicators.macro_momentum.
    Returns a DataFrame indexed by datetime with boolean/int columns and a
    'pos_count' column indicating number of green macros.
    """
    macro_df = compute_macro_indicators(params=MACRO_PARAMS)
    if getattr(macro_df.index, 'tz', None) is not None:
        macro_df.index = macro_df.index.tz_localize(None)
    return macro_df.sort_index()


# --- Backtest core -----------------------------------------------------------

def run_backtest():
    print(f"📈 Running backtest on {TIMEFRAME} data via yfinance for {SYMBOL}...")

    price_df = load_price_data()
    mom_df = compute_momentum_indicators(price_df, params=MOM_PARAMS)
    macro_df = load_macro_data()

    # Align/join safely (both tz-naive) and inner join to overlapping window
    combined = price_df[["close"]].join(mom_df, how="inner").join(macro_df, how="inner")

    # Alignment (e.g., sector breadth). If you don't have sector data yet,
    # compute_alignment can synthesize a simple breadth from momentum signals.
    align_df = compute_alignment(combined, params=ALIGN_PARAMS)
    combined = combined.join(align_df, how="left")

    # --- Entry/Exit rules ----------------------------------------------------
    entry_long_raw = (combined["ema_fast"] > combined["ema_slow"]) & (combined["zscore"] > MOM_PARAMS.z_entry)
    entry_short_raw = (combined["ema_fast"] < combined["ema_slow"]) & (combined["zscore"] < -MOM_PARAMS.z_entry)

    ok_macro = combined["pos_count"] >= MACRO_PARAMS.required_positive
    ok_align = combined["aligned_count"] >= ALIGN_PARAMS.min_aligned

    entries_long = (entry_long_raw & ok_macro & ok_align).astype(int)
    entries_short = (entry_short_raw & ok_macro & ok_align).astype(int)

    exit_long = (combined["ema_fast"] < combined["ema_slow"]) | (combined["zscore"] < MOM_PARAMS.z_exit)
    exit_short = (combined["ema_fast"] > combined["ema_slow"]) | (combined["zscore"] > -MOM_PARAMS.z_exit)

    position = []
    pos = 0
    for t in combined.index:
        if entries_long.loc[t]:
            pos = 1
        elif entries_short.loc[t]:
            pos = -1
        elif pos == 1 and exit_long.loc[t]:
            pos = 0
        elif pos == -1 and exit_short.loc[t]:
            pos = 0
        position.append(pos)
    combined["position"] = position

    entries = (combined["position"].diff().fillna(0) != 0) & (combined["position"] != 0)
    n_long = int(((combined["position"].diff() == 1)).sum())
    n_short = int(((combined["position"].diff() == -1)).sum())

    print("candles:", len(combined))
    print("entries_long:", n_long, "entries_short:", n_short, "total:", n_long + n_short)
    print("macro ok bars:", int(ok_macro.sum()), "align ok bars:", int(ok_align.sum()))

    if n_long + n_short == 0:
        print("⚠️ No trades executed in this backtest window.")
        return

    combined["ret"] = combined["close"].pct_change().fillna(0)
    combined["strategy_ret"] = combined["position"].shift(1).fillna(0) * combined["ret"]
    equity = (1 + combined["strategy_ret"]).cumprod()
    print("Final equity:", round(float(equity.iloc[-1]), 4))


if __name__ == "__main__":
    run_backtest()

