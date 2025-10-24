"""
V51.1 BALANCED HYBRID STRATEGY
================================

Tuned version of V51 with more balanced quality filters.

Changes from V51:
- Relaxed volume spike requirement (1.5x → 1.2x)
- Reduced RSI confirmation period (3 days → 1 day)
- Loosened volatility filter (5% → 7%)
- Removed falling momentum filter (too restrictive)
- Adjusted price ranges

Goal: Find the sweet spot between V48 (no filters) and V51 (too restrictive)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class V51Balanced:
    def __init__(self, initial_capital=30000):
        self.initial_capital = initial_capital
        self.max_positions = 3
        self.position_size_pct = 1.0 / self.max_positions  # 33.33% per position
        
        # Entry/Exit Parameters (from V48)
        self.rsi_entry = 30
        self.rsi_exit = 70
        self.stop_loss_pct = 0.08  # 8%
        self.take_profit_pct = 0.15  # 15%
        
        # BALANCED Quality Filters - Much more lenient
        self.min_volume = 500000  # Reduced from 1M (avoid only very low liquidity)
        self.min_price = 3.0  # Reduced from 5 (allow more stocks)
        self.max_price = 1000.0  # Increased from 500 (don't exclude expensive stocks)
        self.rsi_confirm_period = 1  # Reduced from 3 (just check current RSI is real)
        self.volume_spike_threshold = 1.2  # Reduced from 1.5 (20% above average is enough)
        self.max_volatility = 7.0  # Increased from 5 (allow more volatile stocks)
        # REMOVED: falling momentum check (was too restrictive)
        
        # Data
        self.data = None
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                       'AMD', 'NFLX', 'CRM', 'ADBE', 'INTC', 'CSCO', 'ORCL', 'IBM']
    
    def load_data(self, start_date='2024-01-01', end_date=None):
        """Load and prepare market data"""
        print("📊 Loading market data...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            import yfinance as yf
            print(f"Downloading data for {len(self.symbols)} symbols...")
            self.data = yf.download(self.symbols, start=start_date, end=end_date, 
                                   group_by='ticker', progress=False)
            print(f"✅ Data loaded: {start_date} to {end_date}")
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            raise
    
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
        BALANCED quality filters - Less restrictive than V51
        Returns (bool, str) - (passes, reason if failed)
        """
        try:
            if date not in df.index:
                return False, "Date not in data"
            
            row = df.loc[date]
            
            # Basic sanity checks only
            if row['Close'] < self.min_price:
                return False, f"Price too low: ${row['Close']:.2f}"
            if row['Close'] > self.max_price:
                return False, f"Price too high: ${row['Close']:.2f}"
            
            # Basic volume check (not too strict)
            if row['Volume'] < self.min_volume:
                return False, f"Volume too low: {row['Volume']:,}"
            
            # Gentle volume confirmation (just 20% above average)
            if row['Volume_Ratio'] < self.volume_spike_threshold:
                return False, f"Low relative volume: {row['Volume_Ratio']:.2f}x"
            
            # Very basic RSI check (just verify the signal is real)
            idx = df.index.get_loc(date)
            if idx >= self.rsi_confirm_period:
                recent_rsi = df['RSI'].iloc[idx-self.rsi_confirm_period:idx+1]
                if not any(recent_rsi <= self.rsi_entry + 10):  # Relaxed to +10
                    return False, "RSI not in oversold zone"
            
            # Allow more volatility (7% instead of 5%)
            if row['ATR_Pct'] > self.max_volatility:
                return False, f"Volatility too high: {row['ATR_Pct']:.1f}%"
            
            # REMOVED: Momentum check (was rejecting too many good trades)
            
            return True, "All filters passed"
            
        except Exception as e:
            return False, f"Error checking filters: {str(e)}"
    
    def run_backtest(self):
        """Execute the backtest"""
        print("\n🚀 Running V51.1 Balanced Backtest...")
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
        positions = {}
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
        print("V51.1 BALANCED RESULTS")
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
            rejection_rate = quality_rejected/total_signals*100
            print(f"Rejection Rate:     {rejection_rate:>15.1f}%")
            
            # Provide feedback on rejection rate
            if rejection_rate > 90:
                print(f"⚠️  STILL TOO RESTRICTIVE! (> 90%)")
            elif rejection_rate > 70:
                print(f"⚠️  May be too selective (> 70%)")
            elif rejection_rate < 20:
                print(f"⚠️  Filters may be too loose (< 20%)")
            else:
                print(f"✅ Good balance (20-70%)")
        
        print("\n" + "="*80)
        print("💡 COMPARISON")
        print("="*80)
        print(f"V48 Original:       195.0% CAGR (no filters)")
        print(f"V51 Original:        -5.3% CAGR (filters too strict, 96.9% rejection)")
        print(f"V51.1 Balanced:     {cagr:>5.1f}% CAGR (balanced filters, {rejection_rate:.1f}% rejection)")
        
        v48_diff = cagr - 195
        v51_diff = cagr - (-5.3)
        print(f"\nVs V48:             {v48_diff:+.1f}%")
        print(f"Vs V51:             {v51_diff:+.1f}%")
        
        print(f"\n📊 Win Rate:")
        win_rate = len(winning) / len(trades_df) * 100
        print(f"V51.1 Win Rate:     {win_rate:.1f}%")
        print(f"Target:             50%+ (quality filters improve accuracy)")
        
        # Exit reason breakdown
        print(f"\n📊 Exit Reasons:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            print(f"{reason:20} {count:>5,} ({count/len(trades_df)*100:>5.1f}%)")
        
        # Best/Worst trades
        if len(trades_df) > 0:
            print(f"\n🏆 Best Trade:")
            best = trades_df.loc[trades_df['pnl'].idxmax()]
            print(f"   {best['symbol']}: ${best['pnl']:,.0f} ({best['pnl_pct']:.1f}%) in {best['hold_days']} days")
            
            print(f"\n💀 Worst Trade:")
            worst = trades_df.loc[trades_df['pnl'].idxmin()]
            print(f"   {worst['symbol']}: ${worst['pnl']:,.0f} ({worst['pnl_pct']:.1f}%) in {worst['hold_days']} days")
        
        # Save results
        trades_df.to_csv('v51_balanced_trades.csv', index=False)
        equity_df.to_csv('v51_balanced_equity.csv', index=False)
        
        print(f"\n📁 Files saved:")
        print(f"  • v51_balanced_trades.csv ({len(trades_df)} trades)")
        print(f"  • v51_balanced_equity.csv ({len(equity_df)} data points)")
        
        print("\n" + "="*80)
        print("🎯 NEXT STEPS:")
        if rejection_rate > 80:
            print("  → Filters still too strict - consider V51.2 with even looser filters")
        elif cagr < 100:
            print("  → Performance below target - may need to loosen filters more")
        elif cagr >= 150:
            print("  → Good performance! Consider paper trading")
        else:
            print("  → Moderate performance - evaluate if filters add value")
        print("="*80)
        
        return {
            'trades': trades_df,
            'equity': equity_df,
            'final_equity': final_equity,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trades_df),
            'rejection_rate': rejection_rate
        }


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                   V51.1 BALANCED STRATEGY                             ║
    ║                                                                        ║
    ║  More balanced quality filters - less restrictive than V51            ║
    ║                                                                        ║
    ║  Key Changes:                                                         ║
    ║    • Volume spike: 1.5x → 1.2x (more lenient)                         ║
    ║    • RSI confirm: 3 days → 1 day (faster entry)                       ║
    ║    • Volatility: 5% → 7% (allow more volatile stocks)                 ║
    ║    • Min volume: 1M → 500K (more opportunities)                        ║
    ║    • REMOVED: Momentum falling check (too restrictive)                ║
    ║                                                                        ║
    ║  Goal: 150-200% CAGR with 45%+ win rate                               ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    strategy = V51Balanced(initial_capital=30000)
    strategy.load_data(start_date='2024-01-01')
    results = strategy.run_backtest()
    
    print("\n✅ Backtest Complete!")
