
"""
strategy_v48_fixed_v2.py

Same strategy as before, but the YahooAdapter is now ultra-defensive:
- Handles MultiIndex columns from yfinance (e.g., ('Open','SPY'))
- If multiple tickers slip in, selects the requested ticker's slice via .xs()
- If duplicate top-level names remain, coerces each to a 1-D Series
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import argparse
import math
import sys
from typing import Optional, Dict, Any, List

# -------------------------
# Configs (identical)
# -------------------------

@dataclass
class CostsConfig:
    commission_per_trade: float = 0.00
    slippage_bps: float = 1.5
    borrow_bps_per_year: float = 0.0
    option_premium_k_atr: float = 0.90
    option_theta_bps_per_day: float = 20.0
    option_leverage: float = 4.0

@dataclass
class RiskConfig:
    vol_target_ann: float = 0.12
    max_pos_risk_per_trade: float = 0.005
    atr_stop_mult: float = 1.75
    trail_stop_mult: float = 1.20
    max_daily_drawdown: float = 0.02

@dataclass
class SignalConfig:
    ema_fast: int = 8
    ema_slow: int = 21
    rsi_len: int = 14
    rsi_buy: int = 55
    rsi_sell: int = 45
    macd_fast: int = 12
    macd_slow: int = 26
    macd_sig: int = 9
    adx_len: int = 14
    adx_trend_min: float = 18.0
    band_bps: float = 8.0
    cooldown_days: int = 2
    target_trades_per_week: float = 2.0
    weekly_entry_budget: int = 3

@dataclass
class BacktestConfig:
    start: str = "2018-01-01"
    end: str = "2025-10-16"
    ticker: str = "SPY"
    use_options_proxy: bool = True
    seed: int = 7
    initial_equity: float = 100_000.0
    atr_len: int = 14
    trades_csv: str = "trades_v48.csv"
    equity_csv: str = "equity_curve_v48.csv"

# -------------------------
# Utils
# -------------------------

def _to_series_1d(x, index=None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            x = x.iloc[:, 0]
        else:
            return pd.Series(dtype=float, index=index)
    if isinstance(x, pd.Series):
        s = x
    else:
        arr = np.asarray(x).reshape(-1)
        s = pd.Series(arr, index=index if index is not None and len(index)==arr.shape[0] else None)
    return pd.to_numeric(s, errors="coerce")

# -------------------------
# Data Adapter (robust)
# -------------------------

class YahooAdapter:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_ohlcv(self, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            print("Please install yfinance: pip install yfinance", file=sys.stderr)
            raise

        df = yf.download(self.ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {self.ticker}.")

        # Case 1: MultiIndex (('Open','SPY'), ...)
        if isinstance(df.columns, pd.MultiIndex):
            # If the second level has the ticker, slice it
            lvl_vals = [level.tolist() for level in df.columns.levels]
            try:
                # Prefer selecting by the last level
                if self.ticker in df.columns.get_level_values(-1):
                    df = df.xs(self.ticker, axis=1, level=-1, drop_level=True)
                # else try first level as tickers
                elif self.ticker in df.columns.get_level_values(0):
                    df = df.xs(self.ticker, axis=1, level=0, drop_level=True)
                else:
                    # Fallback: pick the first symbol in the last level
                    sym = df.columns.levels[-1][0]
                    df = df.xs(sym, axis=1, level=-1, drop_level=True)
            except Exception as e:
                # As a final fallback, flatten by taking the top level names,
                # which may leave duplicates; we will coerce each later.
                df.columns = df.columns.get_level_values(0)

        # Normalize names
        df = df.rename(columns=lambda c: str(c).strip().title())

        # At this point, df may still have duplicated column names.
        # Build a 1-D clean frame with only needed columns.
        out = pd.DataFrame(index=df.index)
        for name in ["Open","High","Low","Close","Volume"]:
            if name in df.columns:
                col = df[name]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                out[name] = _to_series_1d(col, getattr(col, "index", None))
            else:
                # Volume might be missing on some assets; default to 0
                if name == "Volume":
                    out[name] = 0
                else:
                    raise ValueError(f"Expected column '{name}' not found after normalization. Got: {list(df.columns)}")

        out = out.dropna(subset=["Open","High","Low","Close"])
        # Ensure dtypes
        out["Open"] = out["Open"].astype(float)
        out["High"] = out["High"].astype(float)
        out["Low"]  = out["Low"].astype(float)
        out["Close"]= out["Close"].astype(float)
        out["Volume"] = out["Volume"].fillna(0).astype(int)

        return out

# -------------------------
# Indicators
# -------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    s = _to_series_1d(series, getattr(series, "index", None))
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    s = _to_series_1d(series, getattr(series, "index", None)).astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int=12, slow: int=26, signal: int=9) -> pd.DataFrame:
    s = _to_series_1d(series, getattr(series, "index", None)).astype(float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})

def true_range(h, l, c_prev):
    return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))

def atr(df: pd.DataFrame, length: int=14) -> pd.Series:
    h = _to_series_1d(df["High"], getattr(df["High"], "index", None))
    l = _to_series_1d(df["Low"], getattr(df["Low"], "index", None))
    c = _to_series_1d(df["Close"], getattr(df["Close"], "index", None))
    c_prev = c.shift(1)
    tr = pd.Series(true_range(h.values, l.values, c_prev.values), index=c.index)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int=14) -> pd.Series:
    high = _to_series_1d(df["High"], getattr(df["High"], "index", None))
    low = _to_series_1d(df["Low"], getattr(df["Low"], "index", None))
    close = _to_series_1d(df["Close"], getattr(df["Close"], "index", None))

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high.values, low.values, close.shift(1).values)
    atr_ = pd.Series(tr, index=close.index).rolling(window=length).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=close.index).rolling(window=length).sum() / (atr_.rolling(window=length).sum() + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).rolling(window=length).sum() / (atr_.rolling(window=length).sum() + 1e-12))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    return dx.rolling(window=length).mean()

# -------------------------
# Strategy + Backtester (same as fixed v1)
# -------------------------

class MomentumAlignmentStrategy:
    def __init__(self, sig: SignalConfig, risk: RiskConfig, costs: CostsConfig, bt: BacktestConfig):
        self.sig = sig
        self.risk = risk
        self.costs = costs
        self.bt = bt

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Close"] = _to_series_1d(out["Close"], getattr(out["Close"], "index", None)).astype(float)

        out["ema_fast"] = ema(out["Close"], self.sig.ema_fast)
        out["ema_slow"] = ema(out["Close"], self.sig.ema_slow)
        out["ema_spread_bps"] = 10_000 * (out["ema_fast"] - out["ema_slow"]) / out["ema_slow"]
        out["RSI"] = rsi(out["Close"], self.sig.rsi_len)

        macd_df = macd(out["Close"], self.sig.macd_fast, self.sig.macd_slow, self.sig.macd_sig)
        out = out.join(macd_df)

        out["ADX"] = adx(out, self.sig.adx_len)
        out["ATR"] = atr(out, self.bt.atr_len)

        z_ema = np.tanh(out["ema_spread_bps"] / (self.sig.band_bps * 2))
        z_rsi = (out["RSI"] - 50) / 25.0
        z_macd = np.tanh(out["hist"] / (out["Close"].rolling(100).std() + 1e-12))
        z_adx = (out["ADX"] - self.sig.adx_trend_min) / 25.0
        out["sig_score"] = 0.35*z_ema + 0.25*z_rsi + 0.25*z_macd + 0.15*z_adx

        out["trend_up"] = (out["ema_spread_bps"] > +self.sig.band_bps)
        out["trend_dn"] = (out["ema_spread_bps"] < -self.sig.band_bps)
        out["rsi_bull"] = (out["RSI"] >= self.sig.rsi_buy)
        out["rsi_bear"] = (out["RSI"] <= self.sig.rsi_sell)
        out["macd_bull"] = (out["macd"] > out["signal"])
        out["macd_bear"] = (out["macd"] < out["signal"])
        out["adx_trendy"] = (out["ADX"] >= self.sig.adx_trend_min)

        out["entry_ok"] = (out["trend_up"] & out["rsi_bull"] & out["macd_bull"] & out["adx_trendy"] & (out["sig_score"] > 0.15))
        out["exit_ok"] = (out["trend_dn"] | out["rsi_bear"] | out["macd_bear"])
        return out

    def position_size(self, equity: float, daily_vol: float) -> float:
        tgt_daily_vol = self.risk.vol_target_ann / math.sqrt(252)
        if daily_vol <= 1e-9:
            return 0.0
        frac = min(tgt_daily_vol / daily_vol, 1.0)
        max_risk_dollars = equity * self.risk.max_pos_risk_per_trade
        units_cap = max_risk_dollars / max(daily_vol, 1e-9)
        return min(frac * equity / 100.0, units_cap)

class Backtester:
    def __init__(self, prices: pd.DataFrame, strat: MomentumAlignmentStrategy, cfg: BacktestConfig, costs: CostsConfig, risk: RiskConfig):
        self.prices = prices.dropna().copy()
        self.strat = strat
        self.cfg = cfg
        self.costs = costs
        self.risk = risk

        self.df = self.strat.build_features(self.prices).dropna().copy()
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def _weekly_key(self, ts: pd.Timestamp) -> str:
        iso = ts.isocalendar()
        return f"{iso[0]}-W{int(iso[1]):02d}"

    def run(self):
        np.random.seed(self.cfg.seed)
        equity = self.cfg.initial_equity
        pos = 0.0
        entry_price = np.nan
        entry_atr = np.nan
        trail_peak = np.nan
        last_exit_day = pd.Timestamp.min
        entries_this_week: Dict[str, int] = {}
        daily_pl = 0.0
        premium_paid = 0.0

        self.df["daily_vol"] = (self.df["ATR"] / self.df["Close"]).clip(lower=1e-6)

        for i, (ts, row) in enumerate(self.df.iterrows()):
            price = float(row["Close"])
            week_key = self._weekly_key(ts)
            if week_key not in entries_this_week:
                entries_this_week[week_key] = 0

            if i > 0:
                prev_ts = self.df.index[i-1]
                if prev_ts.date() != ts.date():
                    daily_pl = 0.0

            if pos != 0.0:
                trail_peak = max(trail_peak, price)

            exit_reason = None

            if pos != 0.0:
                hard_stop = entry_price - self.risk.atr_stop_mult * entry_atr
                trail_stop = trail_peak - self.risk.trail_stop_mult * row["ATR"]

                if self.cfg.use_options_proxy:
                    ret = (price / entry_price - 1.0)
                    gross = self.costs.option_leverage * ret * abs(pos)
                    gross = max(gross, -premium_paid * abs(pos))
                    theta = - (self.costs.option_theta_bps_per_day / 10_000.0) * premium_paid * abs(pos)
                    pnl_today = gross + theta
                else:
                    ret = (price / entry_price - 1.0)
                    pnl_today = ret * abs(pos)

                daily_pl += pnl_today
                guard_hit = (daily_pl / equity) <= -self.risk.max_daily_drawdown
                signal_exit = bool(row["exit_ok"])
                triggered_stop = price <= min(hard_stop, trail_stop)

                if triggered_stop:
                    exit_reason = "stop"
                elif guard_hit:
                    exit_reason = "daily_guard"
                elif signal_exit:
                    exit_reason = "signal_exit"

                if exit_reason is not None:
                    cost = self.costs.commission_per_trade + (self.costs.slippage_bps / 10_000.0) * abs(pos)
                    equity += pnl_today - cost
                    self.trades.append({"time": ts, "type": "EXIT", "price": price, "reason": exit_reason, "pnl": pnl_today - cost, "equity": equity})
                    pos = 0.0
                    entry_price = np.nan
                    entry_atr = np.nan
                    trail_peak = np.nan
                    last_exit_day = ts
                    premium_paid = 0.0

            can_cooldown = (ts - last_exit_day).days >= self.strat.sig.cooldown_days
            can_budget = entries_this_week[week_key] < self.strat.sig.weekly_entry_budget

            if pos == 0.0 and can_cooldown and can_budget and bool(row["entry_ok"])):
                p = min(self.strat.sig.target_trades_per_week / 5.0, 1.0)
                if np.random.rand() < p:
                    size_dollars = max(equity * min(0.25, self.risk.max_pos_risk_per_trade * 15),
                                       self.strat.position_size(equity, row["daily_vol"]) * 10)

                    if size_dollars > 100:
                        cost = self.costs.commission_per_trade + (self.costs.slippage_bps / 10_000.0) * size_dollars
                        if self.cfg.use_options_proxy:
                            premium = self.costs.option_premium_k_atr * row["ATR"]
                            premium_paid = premium
                            units = max(size_dollars / max(premium, 1e-6), 1.0)
                            pos = units
                            equity -= (premium * units)
                            entry_price = price
                            entry_atr = row["ATR"]
                            trail_peak = price
                            self.trades.append({"time": ts, "type": "ENTRY", "price": price, "style": "options_proxy", "units": units, "premium_per_unit": premium, "costs": cost, "equity_after_premium": equity - cost})
                            equity -= cost
                        else:
                            units = size_dollars / price
                            pos = units
                            entry_price = price
                            entry_atr = row["ATR"]
                            trail_peak = price
                            equity -= cost
                            self.trades.append({"time": ts, "type": "ENTRY", "price": price, "style": "underlying", "units": units, "costs": cost, "equity": equity})
                        entries_this_week[week_key] += 1

            if pos != 0.0:
                if self.cfg.use_options_proxy:
                    ret = (price / entry_price - 1.0)
                    gross = self.costs.option_leverage * ret * abs(pos)
                    gross = max(gross, -premium_paid * abs(pos))
                    theta = - (self.costs.option_theta_bps_per_day / 10_000.0) * premium_paid * abs(pos)
                    eq = equity + gross + theta
                else:
                    ret = (price / entry_price - 1.0)
                    eq = equity + ret * abs(pos)
            else:
                eq = equity

            self.equity_curve.append({"time": ts, "price": price, "equity": eq, "pos": pos})

        self.equity_df = pd.DataFrame(self.equity_curve).set_index("time")
        self.trades_df = pd.DataFrame(self.trades)

    def _stats(self) -> Dict[str, Any]:
        df = self.equity_df.copy()
        df["ret"] = df["equity"].pct_change().fillna(0.0)
        total_ret = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0
        yrs = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
        cagr = (1 + total_ret) ** (1/yrs) - 1 if total_ret > -1 else -1.0
        vol = df["ret"].std() * math.sqrt(252)
        sharpe = (df["ret"].mean() * 252) / (vol + 1e-12)
        roll_max = df["equity"].cummax()
        drawdown = df["equity"] / roll_max - 1.0
        max_dd = drawdown.min()
        wins = 0
        losses = 0
        pl_list = []
        if not self.trades_df.empty:
            exits = self.trades_df[self.trades_df["type"] == "EXIT"]
            if "pnl" in exits.columns:
                pl_list = exits["pnl"].tolist()
                wins = sum(1 for x in pl_list if x > 0)
                losses = sum(1 for x in pl_list if x <= 0)
        win_rate = wins / max(wins + losses, 1)
        if not self.trades_df.empty:
            entries = self.trades_df[self.trades_df["type"] == "ENTRY"]
            weeks = max((self.equity_df.index[-1] - self.equity_df.index[0]).days / 7.0, 1e-9)
            trades_per_week = len(entries) / weeks
        else:
            trades_per_week = 0.0
        return {"start_equity": self.cfg.initial_equity,
                "end_equity": float(df["equity"].iloc[-1]),
                "total_return": float(total_ret),
                "CAGR": float(cagr),
                "vol_annual": float(vol),
                "Sharpe": float(sharpe),
                "max_drawdown": float(max_dd),
                "num_trades": int((self.trades_df["type"] == "ENTRY").sum() if not self.trades_df.empty else 0),
                "trades_per_week": float(trades_per_week),
                "win_rate": float(win_rate),
                "avg_realized_pnl": float(np.mean(pl_list)) if pl_list else 0.0}

    def export(self, trades_csv: str, equity_csv: str):
        if not self.trades_df.empty:
            self.trades_df.to_csv(trades_csv, index=False)
        self.equity_df.to_csv(equity_csv)

# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default=BacktestConfig.start)
    p.add_argument("--end", type=str, default=BacktestConfig.end)
    p.add_argument("--ticker", type=str, default=BacktestConfig.ticker)
    p.add_argument("--no-options", action="store_true", help="Trade underlying instead of options proxy")
    p.add_argument("--seed", type=int, default=BacktestConfig.seed)
    return p.parse_args()

def main():
    args = parse_args()
    bt_cfg = BacktestConfig(start=args.start, end=args.end, ticker=args.ticker, use_options_proxy=not args.no_options, seed=args.seed)
    sig = SignalConfig()
    risk = RiskConfig()
    costs = CostsConfig()

    data = YahooAdapter(bt_cfg.ticker).get_ohlcv(bt_cfg.start, bt_cfg.end)
    strat = MomentumAlignmentStrategy(sig, risk, costs, bt_cfg)

    bt = Backtester(data, strat, bt_cfg, costs, risk)
    bt.run()
    stats = bt._stats()
    bt.export(bt_cfg.trades_csv, bt_cfg.equity_csv)

    print("\n=== strategy_v48_fixed_v2 report ===")
    for k, v in stats.items():
        if isinstance(v, float):
            if "rate" in k.lower() or k in ("total_return", "CAGR", "vol_annual", "Sharpe", "max_drawdown", "win_rate", "trades_per_week"):
                print(f"{k:>18}: {v:,.4f}")
            else:
                print(f"{k:>18}: {v:,.2f}")
        else:
            print(f"{k:>18}: {v}")
    print(f"\nTrades CSV : {bt_cfg.trades_csv}")
    print(f"Equity CSV : {bt_cfg.equity_csv}")
    print("==============\n")

if __name__ == "__main__":
    main()
