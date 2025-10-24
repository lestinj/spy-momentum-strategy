"""
V51 HYBRID TRADING STRATEGY
============================

Combines the best of both worlds:
- V48's proven setup: 195% CAGR with 3 positions, static allocation
- V50's quality filters: Improved win rate and trade selection

Goal: Match or exceed V48's performance with better win rate and fewer bad trades

Key Features:
- 3 simultaneous positions (proven sweet spot)
- Static position sizing (33.33% per position)
- Quality filters for better trade selection
- RSI and volume confirmation
- Trend strength validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class V51Hybrid:
    def __init__(self, initial_capital=30000):
        self.initial_capital = initial_capital
        self.max_positions = 3
        self.position_size_pct = 1.0 / self.max_positions  # 33.33% per position
        
        # Entry/Exit Parameters (from V48)
        self.rsi_entry = 30
        self.rsi_exit = 70
        self.stop_loss_pct = 0.08  # 8%
        self.take_profit_pct = 0.15  # 15%
        
        # Quality Filters (from V50)
        self.min_volume = 1000000  # Minimum daily volume
        self.min_price = 5.0  # Avoid penny stocks
        self.max_price = 500.0  # Reasonable price range
        self.rsi_confirm_period = 3  # Days to confirm RSI trend
        self.volume_spike_threshold = 1.5  # Volume 50% above average
        
        # Data
        self.data = None
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                       'AMD', 'NFLX', 'CRM', 'ADBE', 'INTC', 'CSCO', 'ORCL', 'IBM']
    
    def load_data(self, start_date='2024-01-01', end_date=None):
        """Load and prepare market data"""
        print("📊 Loading market data...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Simulate data loading (replace with actual data source)
        try:
            import yfinance as yf
            print(f"Downloading data for {len(self.symbols)} symbols...")
            self.data = yf.download(self.symbols, start=start_date, end=end_date, 
                                   group_by='ticker', progress=False)
            print(f"✅ Data loaded: {start_date} to {end_date}")
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            print("Using synthetic data for demonstration...")
            self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, start_date, end_date):
        """Generate synthetic data for testing"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data_dict = {}
        for symbol in self.symbols:
            np.random.seed(hash(symbol) % 2**32)
            
            # Generate realistic price movements
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Add some volatility clusters
            volatility = np.random.uniform(0.5, 2.0, len(dates))
            prices = prices * (1 + np.random.normal(0, 0.01, len(dates)) * volatility)
            
            data_dict[symbol] = pd.DataFrame({
                'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
                'High': prices * np.random.uniform(1.00, 1.05, len(dates)),
                'Low': prices * np.random.uniform(0.95, 1.00, len(dates)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        
        self.data = data_dict
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['Price_Change_5d'] = df['Close'].pct_change(5) * 100
        df['Price_Change_20d'] = df['Close'].pct_change(20) * 100
        
        # Volatility (ATR approximation)
        df['High_Low'] = df['High'] - df['Low']
        df['ATR'] = df['High_Low'].rolling(window=14).mean()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        return df
    
    def check_quality_filters(self, symbol, date, df):
        """
        Check if a potential trade meets quality criteria
        Returns (bool, str) - (passes, reason if failed)
        """
        try:
            if date not in df.index:
                return False, "Date not in data"
            
            row = df.loc[date]
            
            # Price range filter
            if row['Close'] < self.min_price:
                return False, f"Price too low: ${row['Close']:.2f}"
            if row['Close'] > self.max_price:
                return False, f"Price too high: ${row['Close']:.2f}"
            
            # Volume filter
            if row['Volume'] < self.min_volume:
                return False, f"Volume too low: {row['Volume']:,}"
            
            # Volume spike confirmation (indicates momentum)
            if row['Volume_Ratio'] < self.volume_spike_threshold:
                return False, f"No volume spike: {row['Volume_Ratio']:.2f}x"
            
            # RSI confirmation (check it's been oversold for a few days)
            idx = df.index.get_loc(date)
            if idx >= self.rsi_confirm_period:
                recent_rsi = df['RSI'].iloc[idx-self.rsi_confirm_period:idx+1]
                if not all(recent_rsi <= self.rsi_entry + 5):
                    return False, "RSI not consistently oversold"
            
            # Avoid extreme volatility (likely to hit stops)
            if row['ATR_Pct'] > 5.0:
                return False, f"Volatility too high: {row['ATR_Pct']:.1f}%"
            
            # Check for positive momentum (starting to recover)
            if row['Price_Change_5d'] < -10:
                return False, f"Still falling: {row['Price_Change_5d']:.1f}%"
            
            return True, "All filters passed"
            
        except Exception as e:
            return False, f"Error checking filters: {str(e)}"
    
    def run_backtest(self):
        """Execute the backtest"""
        print("\n🚀 Running V51 Hybrid Backtest...")
        print("=" * 80)
        
        # Prepare data
        all_data = {}
        for symbol in self.symbols:
            if isinstance(self.data, dict):
                df = self.data[symbol].copy()
            else:
                df = self.data[symbol].copy()
            
            df = self.calculate_indicators(df)
            all_data[symbol] = df
        
        # Get common date range
        all_dates = set(all_data[self.symbols[0]].index)
        for symbol in self.symbols[1:]:
            all_dates &= set(all_data[symbol].index)
        all_dates = sorted(list(all_dates))
        
        # Initialize tracking
        equity = self.initial_capital
        positions = {}  # {symbol: {'entry_price', 'entry_date', 'shares', 'stop', 'target'}}
        trades = []
        equity_curve = []
        
        quality_rejected = 0
        quality_accepted = 0
        
        print(f"📅 Backtesting from {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.0f}")
        print(f"🎯 Max Positions: {self.max_positions}")
        print(f"📊 Position Size: {self.position_size_pct*100:.1f}% per trade")
        print()
        
        # Main backtest loop
        for i, date in enumerate(all_dates):
            # Check exits first
            positions_to_close = []
            
            for symbol, pos in positions.items():
                current_price = all_data[symbol].loc[date, 'Close']
                current_rsi = all_data[symbol].loc[date, 'RSI']
                
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                
                # Exit conditions
                exit_reason = None
                if current_price <= pos['stop']:
                    exit_reason = 'Stop Loss'
                elif current_price >= pos['target']:
                    exit_reason = 'Take Profit'
                elif current_rsi >= self.rsi_exit:
                    exit_reason = 'RSI Exit'
                
                if exit_reason:
                    # Close position
                    exit_value = pos['shares'] * current_price
                    pnl = exit_value - (pos['shares'] * pos['entry_price'])
                    
                    equity += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'entry_price': pos['entry_price'],
                        'exit_date': date,
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': exit_reason,
                        'hold_days': (date - pos['entry_date']).days
                    })
                    
                    positions_to_close.append(symbol)
            
            # Remove closed positions
            for symbol in positions_to_close:
                del positions[symbol]
            
            # Look for new entries (if we have room)
            if len(positions) < self.max_positions:
                # Check all symbols for entry signals
                entry_candidates = []
                
                for symbol in self.symbols:
                    # Skip if already in position
                    if symbol in positions:
                        continue
                    
                    df = all_data[symbol]
                    current_price = df.loc[date, 'Close']
                    current_rsi = df.loc[date, 'RSI']
                    
                    # Entry signal: RSI oversold
                    if current_rsi <= self.rsi_entry:
                        # Check quality filters
                        passes_filter, reason = self.check_quality_filters(symbol, date, df)
                        
                        if passes_filter:
                            # Calculate a quality score for ranking
                            volume_score = df.loc[date, 'Volume_Ratio']
                            rsi_score = self.rsi_entry - current_rsi  # Lower RSI = higher score
                            momentum_score = df.loc[date, 'Price_Change_5d'] if df.loc[date, 'Price_Change_5d'] > 0 else 0
                            
                            quality_score = volume_score + (rsi_score / 10) + (momentum_score / 10)
                            
                            entry_candidates.append({
                                'symbol': symbol,
                                'price': current_price,
                                'rsi': current_rsi,
                                'quality_score': quality_score
                            })
                            quality_accepted += 1
                        else:
                            quality_rejected += 1
                
                # Take the best quality signals up to our position limit
                entry_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
                slots_available = self.max_positions - len(positions)
                
                for candidate in entry_candidates[:slots_available]:
                    symbol = candidate['symbol']
                    entry_price = candidate['price']
                    
                    # Calculate position size
                    position_value = equity * self.position_size_pct
                    shares = int(position_value / entry_price)
                    
                    if shares > 0:
                        # Open position
                        positions[symbol] = {
                            'entry_price': entry_price,
                            'entry_date': date,
                            'shares': shares,
                            'stop': entry_price * (1 - self.stop_loss_pct),
                            'target': entry_price * (1 + self.take_profit_pct)
                        }
            
            # Record equity
            positions_value = sum(pos['shares'] * all_data[symbol].loc[date, 'Close'] 
                                 for symbol, pos in positions.items())
            cash = equity - sum(pos['shares'] * pos['entry_price'] for pos in positions.values())
            total_equity = cash + positions_value
            
            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'positions': len(positions),
                'cash': cash
            })
        
        # Close any remaining positions at the end
        final_date = all_dates[-1]
        for symbol, pos in positions.items():
            current_price = all_data[symbol].loc[final_date, 'Close']
            exit_value = pos['shares'] * current_price
            pnl = exit_value - (pos['shares'] * pos['entry_price'])
            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            equity += pnl
            
            trades.append({
                'symbol': symbol,
                'entry_date': pos['entry_date'],
                'entry_price': pos['entry_price'],
                'exit_date': final_date,
                'exit_price': current_price,
                'shares': pos['shares'],
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'exit_reason': 'End of Backtest',
                'hold_days': (final_date - pos['entry_date']).days
            })
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        cagr = (((final_equity / self.initial_capital) ** (1 / years)) - 1) * 100
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = ((equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']) * 100
        max_dd = equity_df['drawdown'].min()
        
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        
        # Print Results
        print("\n" + "="*80)
        print("V51 HYBRID RESULTS")
        print("="*80)
        print(f"Initial Capital:    ${self.initial_capital:>15,.0f}")
        print(f"Final Equity:       ${final_equity:>15,.0f}")
        print(f"Total Return:       {total_return:>15.1f}%")
        print(f"CAGR:               {cagr:>15.1f}%")
        print(f"Max Drawdown:       {max_dd:>15.1f}%")
        print(f"\nTotal Trades:       {len(trades_df):>15,}")
        print(f"Winners:            {len(winning):>15,} ({len(winning)/len(trades_df)*100:.1f}%)")
        print(f"Losers:             {len(losing):>15,} ({len(losing)/len(trades_df)*100:.1f}%)")
        print(f"Avg Win:            ${winning['pnl'].mean():>14,.0f}")
        print(f"Avg Loss:           ${losing['pnl'].mean():>14,.0f}" if len(losing) > 0 else "Avg Loss:           $0")
        
        if len(losing) > 0 and losing['pnl'].sum() != 0:
            pf = abs(winning['pnl'].sum() / losing['pnl'].sum())
            print(f"Profit Factor:      {pf:>15.2f}")
        
        avg_hold = trades_df['hold_days'].mean()
        print(f"Avg Hold Days:      {avg_hold:>15.1f}")
        
        print("\n" + "="*80)
        print("QUALITY FILTER IMPACT")
        print("="*80)
        print(f"Signals Accepted:   {quality_accepted:>15,}")
        print(f"Signals Rejected:   {quality_rejected:>15,}")
        total_signals = quality_accepted + quality_rejected
        if total_signals > 0:
            print(f"Rejection Rate:     {quality_rejected/total_signals*100:>15.1f}%")
        
        print("\n" + "="*80)
        print("💡 COMPARISON")
        print("="*80)
        print(f"V48 Original:       195.0% CAGR (3 pos, static, no filters)")
        print(f"V51 Hybrid:         {cagr:>5.1f}% CAGR (3 pos, static, WITH filters)")
        print(f"Difference:         {cagr - 195:.1f}%")
        
        print(f"\n📊 Win Rate Comparison:")
        win_rate = len(winning) / len(trades_df) * 100
        print(f"V51 Win Rate:       {win_rate:.1f}%")
        print(f"Target:             50%+ (quality filters improve accuracy)")
        
        # Exit reason breakdown
        print(f"\n📊 Exit Reasons:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            print(f"{reason:20} {count:>5,} ({count/len(trades_df)*100:>5.1f}%)")
        
        # Best/Worst trades
        print(f"\n🏆 Best Trade:")
        best = trades_df.loc[trades_df['pnl'].idxmax()]
        print(f"   {best['symbol']}: ${best['pnl']:,.0f} ({best['pnl_pct']:.1f}%) in {best['hold_days']} days")
        
        print(f"\n💀 Worst Trade:")
        worst = trades_df.loc[trades_df['pnl'].idxmin()]
        print(f"   {worst['symbol']}: ${worst['pnl']:,.0f} ({worst['pnl_pct']:.1f}%) in {worst['hold_days']} days")
        
        # Save results
        trades_df.to_csv('v51_hybrid_trades.csv', index=False)
        equity_df.to_csv('v51_hybrid_equity.csv', index=False)
        
        print(f"\n📁 Files saved:")
        print(f"  • v51_hybrid_trades.csv ({len(trades_df)} trades)")
        print(f"  • v51_hybrid_equity.csv ({len(equity_df)} data points)")
        
        print("\n" + "="*80)
        
        return {
            'trades': trades_df,
            'equity': equity_df,
            'final_equity': final_equity,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades_df)
        }


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                      V51 HYBRID STRATEGY                              ║
    ║                                                                        ║
    ║  Combining V48's proven performance with V50's quality filters        ║
    ║                                                                        ║
    ║  Strategy:                                                            ║
    ║    • 3 simultaneous positions (proven optimal)                        ║
    ║    • Static 33.33% allocation per position                            ║
    ║    • RSI-based entry (RSI ≤ 30)                                       ║
    ║    • Quality filters for trade selection                              ║
    ║    • 8% stop loss, 15% take profit                                    ║
    ║                                                                        ║
    ║  Goal: 195%+ CAGR with 50%+ win rate                                  ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    strategy = V51Hybrid(initial_capital=30000)
    strategy.load_data(start_date='2024-01-01')
    results = strategy.run_backtest()
    
    print("\n✅ Backtest Complete!")
    print("\n💡 Next Steps:")
    print("  1. Review the results above")
    print("  2. Check v51_hybrid_trades.csv for detailed trade history")
    print("  3. Check v51_hybrid_equity.csv for equity curve")
    print("  4. Compare with V48 baseline (195% CAGR)")
    print("  5. If results are promising, consider paper trading")
    print("\n🎯 Success Criteria:")
    print("  • CAGR ≥ 195% (match or beat V48)")
    print("  • Win Rate ≥ 50% (quality improvement)")
    print("  • Max Drawdown < 30%")
    print("  • Profit Factor > 2.0")
