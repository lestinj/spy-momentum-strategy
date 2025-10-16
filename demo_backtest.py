"""
DEMO VERSION - Simulated Data Backtest
Works without network access - generates synthetic SPY data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import sys
sys.path.insert(0, '/home/claude/spy-momentum-strategy')

from indicators.momentum import compute_momentum_indicators, MomentumParams
from indicators.alignment_checker import compute_alignment, AlignmentParams


def generate_synthetic_spy_data(start_date: str, end_date: str, freq='1H') -> pd.DataFrame:
    """
    Generate realistic synthetic SPY data for testing
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create hourly timestamps (trading hours only)
    dates = pd.date_range(start, end, freq=freq)
    # Filter to trading hours (9:30 - 16:00)
    dates = dates[(dates.hour >= 9) & (dates.hour <= 16)]
    
    n = len(dates)
    
    # Generate price with trend + noise
    base_price = 570  # SPY base price
    trend = np.linspace(0, 15, n)  # Upward trend
    cycles = 10 * np.sin(np.linspace(0, 8 * np.pi, n))  # Cyclical movement
    noise = np.random.normal(0, 2, n)  # Random noise
    
    close = base_price + trend + cycles + noise
    
    # Generate OHLC from close
    high = close + np.random.uniform(0.5, 2, n)
    low = close - np.random.uniform(0.5, 2, n)
    open_price = close + np.random.uniform(-1, 1, n)
    volume = np.random.randint(5000000, 15000000, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def generate_synthetic_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate synthetic macro indicators
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    dates = pd.date_range(start, end, freq='D')
    n = len(dates)
    
    # Generate indicators with trends
    df = pd.DataFrame({
        'consumer_sentiment': 65 + 5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 2, n),
        'unemployment_rate': 4.0 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(0, 0.1, n),
        'yield_curve': 0.3 + 0.2 * np.sin(np.linspace(0, 5 * np.pi, n)) + np.random.normal(0, 0.1, n),
        'vix': 18 + 8 * np.abs(np.sin(np.linspace(0, 6 * np.pi, n))) + np.random.normal(0, 1, n),
    }, index=dates)
    
    # Generate bullish signals
    df['consumer_bullish'] = df['consumer_sentiment'] > 65
    df['unemployment_improving'] = df['unemployment_rate'] < 4.2
    df['yield_curve_positive'] = df['yield_curve'] > 0
    df['vix_low'] = df['vix'] < 20
    df['industrial_growth'] = np.random.choice([True, False], n, p=[0.6, 0.4])
    df['oil_stable'] = np.random.choice([True, False], n, p=[0.7, 0.3])
    
    signal_cols = ['consumer_bullish', 'unemployment_improving', 'yield_curve_positive', 
                   'vix_low', 'industrial_growth', 'oil_stable']
    df['positive_macro_count'] = df[signal_cols].sum(axis=1)
    df['macro_bullish'] = df['positive_macro_count'] >= 3
    
    return df


class DemoBacktest:
    """Demo backtest using synthetic data"""
    
    def __init__(self):
        self.initial_capital = 100000
        self.options_multiplier = 100
        
        # Tuned for 2+ trades per week
        self.mom_params = MomentumParams(
            fast=10,
            slow=30,
            rsi_len=7,
            rsi_buy=55.0,
            rsi_sell=45.0,
            z_window=10,
            z_entry=0.3,
            z_exit=-0.1,
        )
        
        self.align_params = AlignmentParams(
            min_aligned=5,
            lookback=5
        )
    
    def run(self, start_date="2024-06-01", end_date="2024-10-16"):
        print(f"\n{'='*70}")
        print(f"DEMO BACKTEST - Using Synthetic Data")
        print(f"{'='*70}\n")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}\n")
        
        # Generate data
        print("Generating synthetic SPY data...")
        price_df = generate_synthetic_spy_data(start_date, end_date, freq='1H')
        print(f"✓ Generated {len(price_df)} bars of price data")
        
        print("Generating synthetic macro data...")
        macro_df = generate_synthetic_macro_data(start_date, end_date)
        print(f"✓ Generated {len(macro_df)} days of macro data\n")
        
        # Compute momentum
        print("Computing momentum indicators...")
        mom_df = compute_momentum_indicators(price_df, self.mom_params)
        
        # Combine datasets
        combined = price_df[['close']].copy()
        combined = combined.join(mom_df, how='inner')
        
        # Add macro (forward fill)
        macro_reindexed = macro_df.reindex(combined.index, method='ffill')
        combined['positive_macro_count'] = macro_reindexed['positive_macro_count'].fillna(2)
        
        # Compute alignment
        align_df = compute_alignment(combined, self.align_params)
        combined = combined.join(align_df, how='left')
        combined['aligned_count'] = combined['aligned_count'].fillna(0)
        
        # Generate signals
        print("Generating signals...")
        combined = self.generate_signals(combined)
        
        # Simulate trades
        print("Simulating trades...")
        result_df, trades = self.simulate_trades(combined)
        
        # Calculate metrics
        metrics = self.calculate_metrics(result_df, trades)
        
        # Print results
        self.print_results(metrics, trades, result_df)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'data': result_df
        }
    
    def generate_signals(self, df):
        """Generate entry/exit signals"""
        df = df.copy()
        
        # Entry conditions
        bullish = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['rsi'] > self.mom_params.rsi_buy) &
            (df['zscore'] > self.mom_params.z_entry)
        )
        
        bearish = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['rsi'] < self.mom_params.rsi_sell) &
            (df['zscore'] < -self.mom_params.z_entry)
        )
        
        macro_ok = df['positive_macro_count'] >= 2
        align_ok = df['aligned_count'] >= self.align_params.min_aligned
        
        df['entry_long'] = bullish & macro_ok & align_ok
        df['entry_short'] = bearish & macro_ok & align_ok
        
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
    
    def simulate_trades(self, df):
        """Simulate trading"""
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        capital = self.initial_capital
        positions = []
        equity_curve = []
        
        for idx, row in df.iterrows():
            # Exit logic
            if position == 1 and row['exit_long']:
                pnl = (row['close'] - entry_price) * self.options_multiplier
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
                pnl = (entry_price - row['close']) * self.options_multiplier
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
            
            # Entry logic
            if position == 0:
                if row['entry_long']:
                    position = 1
                    entry_price = row['close']
                    entry_date = idx
                elif row['entry_short']:
                    position = -1
                    entry_price = row['close']
                    entry_date = idx
            
            positions.append(position)
            
            if position != 0:
                current_pnl = (row['close'] - entry_price) * position * self.options_multiplier
                current_equity = capital + current_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        df['position'] = positions
        df['equity'] = equity_curve
        
        return df, trades
    
    def calculate_metrics(self, df, trades):
        """Calculate performance metrics"""
        if not trades:
            return {'total_trades': 0, 'trades_per_week': 0}
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        weeks = (df.index[-1] - df.index[0]).days / 7
        trades_per_week = total_trades / weeks if weeks > 0 else 0
        
        returns = df['equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        equity = df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
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
    
    def print_results(self, metrics, trades, df):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*70}\n")
        
        if metrics['total_trades'] == 0:
            print("⚠️  No trades generated. Try adjusting parameters.")
            return
        
        print(f"Trading Performance:")
        print(f"  Total Trades:        {metrics['total_trades']}")
        print(f"  Long Trades:         {metrics['long_trades']}")
        print(f"  Short Trades:        {metrics['short_trades']}")
        print(f"  Trades per Week:     {metrics['trades_per_week']:.2f} ⭐")
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
        
        print(f"\n{'='*70}")
        
        # Show sample trades
        if trades:
            print("\nSample Trades (First 5):")
            trades_df = pd.DataFrame(trades)
            sample = trades_df.head(5)
            for _, trade in sample.iterrows():
                pnl_indicator = "✓" if trade['pnl'] > 0 else "✗"
                print(f"  {pnl_indicator} {trade['entry_date'].strftime('%Y-%m-%d %H:%M')} "
                      f"{trade['direction']:5s} ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                      f"P&L: ${trade['pnl']:,.2f}")
        
        print(f"\n{'='*70}")
        print(f"✨ Demo completed successfully!")
        print(f"{'='*70}\n")
        
        # Equity curve summary
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        print(f"Equity Curve:")
        print(f"  Start:  ${initial_equity:,.2f}")
        print(f"  End:    ${final_equity:,.2f}")
        print(f"  Change: ${final_equity - initial_equity:,.2f} ({(final_equity/initial_equity - 1)*100:.2f}%)")
        print()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMO BACKTEST - SPY MOMENTUM STRATEGY")
    print("="*70)
    print("\nThis demo uses SYNTHETIC DATA to demonstrate the strategy logic.")
    print("For real backtests with actual market data, ensure network access")
    print("and use improved_backtest.py instead.\n")
    
    backtest = DemoBacktest()
    results = backtest.run(start_date="2024-06-01", end_date="2024-10-16")
    
    print("\n📊 Summary:")
    if results['metrics']['trades_per_week'] >= 2.0:
        print(f"✅ TARGET MET: {results['metrics']['trades_per_week']:.2f} trades/week (goal: 2+)")
    else:
        print(f"⚠️  Below target: {results['metrics']['trades_per_week']:.2f} trades/week")
        print(f"   Consider relaxing parameters for more trades")
    
    print("\n💡 Next Steps:")
    print("  1. Adjust parameters in the code to tune performance")
    print("  2. Test with real data using improved_backtest.py (requires network)")
    print("  3. Use improved_main.py for live trading signals")
    print()
