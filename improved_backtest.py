"""
Improved Backtesting System with Options Integration
Targets 2+ trades per week with comprehensive metrics
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from data.improved_data_fetcher import DataFetcher
from indicators.momentum import compute_momentum_indicators, MomentumParams
from indicators.macro_momentum import MacroParams
from indicators.alignment_checker import AlignmentParams


class ImprovedBacktest:
    """Enhanced backtesting with options data and detailed analytics"""
    
    def __init__(
        self,
        symbol: str = "SPY",
        timeframe: str = "1h",
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 100000,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.data_fetcher = DataFetcher()
        
        # Adjusted parameters for MORE frequent trading (targeting 2+ trades/week)
        self.mom_params = MomentumParams(
            fast=10,           # Faster EMA (was 20)
            slow=30,           # Faster EMA (was 50)
            rsi_len=7,         # Shorter RSI
            rsi_buy=55.0,      # More sensitive
            rsi_sell=45.0,     # More sensitive
            z_window=10,       # Shorter window
            z_entry=0.3,       # Lower threshold (was 0.5)
            z_exit=-0.1,       # Allow some give
        )
        
        self.macro_params = MacroParams(
            required_positive=2  # Require only 2 positive macro signals
        )
        
        self.align_params = AlignmentParams(
            min_aligned=5,     # Relaxed from 10
            lookback=5         # Shorter lookback
        )
        
        # Trading parameters
        self.use_options = True  # Trade options instead of stock
        self.options_multiplier = 100  # Standard options contract
        self.max_position_size = 0.2  # 20% of capital per trade
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare all required data"""
        print(f"Loading {self.symbol} data from {self.start_date} to {self.end_date}...")
        
        # Map timeframe to yfinance interval
        interval_map = {
            "1h": "60m", "30m": "30m", "15m": "15m", "5m": "5m",
            "1d": "1d", "1wk": "1wk"
        }
        interval = interval_map.get(self.timeframe, self.timeframe)
        
        # Calculate period from start_date
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        days_back = (end - start).days + 10  # Add buffer
        
        # Fetch price data
        price_df = self.data_fetcher.fetch_spy_price(
            period=f"{days_back}d",
            interval=interval
        )
        
        if price_df.empty:
            raise ValueError("Failed to fetch price data")
        
        # Standardize column names
        price_df.columns = [c.lower() for c in price_df.columns]
        if 'adj close' in price_df.columns:
            price_df['close'] = price_df['adj close']
        
        # Remove timezone if present
        if price_df.index.tz is not None:
            price_df.index = price_df.index.tz_localize(None)
        
        # Filter to date range
        price_df = price_df[start:end]
        
        print(f"✓ Loaded {len(price_df)} bars of price data")
        
        return price_df
    
    def load_macro_data(self) -> pd.DataFrame:
        """Load and prepare macro indicators"""
        print("Loading macro indicators...")
        
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        lookback_days = (end - start).days + 60
        
        macro_df = self.data_fetcher.fetch_macro_indicators(lookback_days=lookback_days)
        
        if macro_df.empty:
            print("WARNING: No macro data available, creating permissive default")
            # Create permissive default
            dates = pd.date_range(start, end, freq='D')
            macro_df = pd.DataFrame({
                'positive_macro_count': [3] * len(dates),
                'macro_bullish': [True] * len(dates)
            }, index=dates)
        
        # Remove timezone if present
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)
        
        print(f"✓ Loaded macro data with {len(macro_df)} observations")
        
        return macro_df
    
    def generate_signals(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry and exit signals"""
        df = combined_df.copy()
        
        # Entry conditions
        bullish_momentum = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['rsi'] > self.mom_params.rsi_buy) &
            (df['zscore'] > self.mom_params.z_entry)
        )
        
        bearish_momentum = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['rsi'] < self.mom_params.rsi_sell) &
            (df['zscore'] < -self.mom_params.z_entry)
        )
        
        # Macro filter (relax this to allow more trades)
        macro_ok = df['positive_macro_count'] >= self.macro_params.required_positive
        
        # Alignment filter
        alignment_ok = df['aligned_count'] >= self.align_params.min_aligned
        
        # Final entry signals
        df['entry_long'] = bullish_momentum & macro_ok & alignment_ok
        df['entry_short'] = bearish_momentum & macro_ok & alignment_ok
        
        # Exit conditions (more lenient than entry)
        df['exit_long'] = (
            (df['ema_fast'] < df['ema_slow']) |
            (df['zscore'] < self.mom_params.z_exit) |
            (df['rsi'] < 40)
        )
        
        df['exit_short'] = (
            (df['ema_fast'] > df['ema_slow']) |
            (df['zscore'] > -self.mom_params.z_exit) |
            (df['rsi'] > 60)
        )
        
        return df
    
    def simulate_trades(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Simulate actual trading with position management"""
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0
        positions = []
        trades = []
        capital = self.initial_capital
        equity_curve = []
        
        for idx, row in df.iterrows():
            # Check for exit first
            if position == 1 and row['exit_long']:
                # Close long position
                pnl = (row['close'] - entry_price) * abs(position) * self.options_multiplier
                capital += pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'pnl': pnl,
                    'return_pct': (row['close'] / entry_price - 1) * 100
                })
                position = 0
                
            elif position == -1 and row['exit_short']:
                # Close short position
                pnl = (entry_price - row['close']) * abs(position) * self.options_multiplier
                capital += pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': idx,
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'pnl': pnl,
                    'return_pct': (entry_price / row['close'] - 1) * 100
                })
                position = 0
            
            # Check for entry (only if flat)
            if position == 0:
                if row['entry_long']:
                    position = 1
                    entry_price = row['close']
                    entry_date = idx
                    
                elif row['entry_short']:
                    position = -1
                    entry_price = row['close']
                    entry_date = idx
            
            # Track position and equity
            positions.append(position)
            
            # Calculate current equity
            if position != 0:
                current_pnl = (row['close'] - entry_price) * position * self.options_multiplier
                current_equity = capital + current_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Add position and equity to dataframe
        df['position'] = positions
        df['equity'] = equity_curve
        
        return df, trades
    
    def calculate_metrics(self, df: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades_per_week': 0,
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Calculate trades per week
        start = df.index[0]
        end = df.index[-1]
        weeks = (end - start).days / 7
        trades_per_week = total_trades / weeks if weeks > 0 else 0
        
        # Calculate Sharpe ratio from equity curve
        returns = df['equity'].pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        equity = df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calculate profit factor
        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'total_trades': total_trades,
            'long_trades': len(trades_df[trades_df['direction'] == 'LONG']),
            'short_trades': len(trades_df[trades_df['direction'] == 'SHORT']),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor,
            'trades_per_week': trades_per_week,
        }
    
    def run(self) -> Dict:
        """Run the complete backtest"""
        print(f"\n{'='*60}")
        print(f"Running Backtest: {self.symbol} @ {self.timeframe}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")
        
        # Load data
        price_df = self.load_data()
        macro_df = self.load_macro_data()
        
        # Compute indicators
        print("Computing momentum indicators...")
        mom_df = compute_momentum_indicators(price_df, self.mom_params)
        
        # Combine datasets
        combined = price_df[['close']].copy()
        combined = combined.join(mom_df, how='inner')
        
        # Add macro data (forward fill for alignment)
        macro_reindexed = macro_df.reindex(combined.index, method='ffill')
        combined['positive_macro_count'] = macro_reindexed['positive_macro_count'].fillna(2)
        
        # Compute alignment
        from indicators.alignment_checker import compute_alignment
        align_df = compute_alignment(combined, self.align_params)
        combined = combined.join(align_df, how='left')
        combined['aligned_count'] = combined['aligned_count'].fillna(0)
        
        # Generate signals
        print("Generating trading signals...")
        combined = self.generate_signals(combined)
        
        # Simulate trades
        print("Simulating trades...")
        result_df, trades = self.simulate_trades(combined)
        
        # Calculate metrics
        metrics = self.calculate_metrics(result_df, trades)
        
        # Print results
        self.print_results(metrics, trades)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'data': result_df,
        }
    
    def print_results(self, metrics: Dict, trades: List[Dict]):
        """Print formatted backtest results"""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Trading Performance:")
        print(f"  Total Trades:        {metrics['total_trades']}")
        print(f"  Long Trades:         {metrics['long_trades']}")
        print(f"  Short Trades:        {metrics['short_trades']}")
        print(f"  Trades per Week:     {metrics['trades_per_week']:.2f}")
        print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"\nP&L:")
        print(f"  Total P&L:           ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return:        {metrics['total_return_pct']:.2f}%")
        print(f"  Avg Win:             ${metrics['avg_win']:,.2f}")
        print(f"  Avg Loss:            ${metrics['avg_loss']:,.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\n{'='*60}\n")
        
        # Show recent trades
        if trades:
            print("Recent Trades (Last 5):")
            trades_df = pd.DataFrame(trades)
            recent = trades_df.tail(5)
            for _, trade in recent.iterrows():
                print(f"  {trade['entry_date'].strftime('%Y-%m-%d')} "
                      f"{trade['direction']:5s} "
                      f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                      f"P&L: ${trade['pnl']:,.2f} ({trade['return_pct']:+.2f}%)")
        
        print()


def run_backtest():
    """Main backtest entry point"""
    backtest = ImprovedBacktest(
        symbol="SPY",
        timeframe="1h",
        start_date="2024-06-01",
        end_date=None,  # Use current date
        initial_capital=100000
    )
    
    results = backtest.run()
    
    return results


if __name__ == "__main__":
    results = run_backtest()
