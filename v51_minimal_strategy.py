"""
V51.2 MINIMAL FILTERS STRATEGY
================================

Final attempt before reverting to V48.
Only the most basic safety filters - almost like V48 but with minimal quality gates.

Changes from V51.1:
- Volume spike requirement REMOVED (was blocking too much)
- Volatility filter REMOVED (was blocking too much)
- RSI confirmation REMOVED (just trust the signal)
- Only keeping: Basic price range and absolute minimum volume

Philosophy: Let most trades through, only block obvious garbage
Target: Get close to V48's 195% CAGR while filtering true junk
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class V51Minimal:
    def __init__(self, initial_capital=30000):
        self.initial_capital = initial_capital
        self.max_positions = 3
        self.position_size_pct = 1.0 / self.max_positions  # 33.33% per position
        
        # Entry/Exit Parameters (from V48)
        self.rsi_entry = 30
        self.rsi_exit = 70
        self.stop_loss_pct = 0.08  # 8%
        self.take_profit_pct = 0.15  # 15%
        
        # MINIMAL Safety Filters - Only block true garbage
        self.min_volume = 100000  # Only block extremely illiquid (reduced from 500K)
        self.min_price = 2.0  # Only block penny stocks (reduced from 3)
        self.max_price = 10000.0  # Essentially no limit (increased from 1000)
        # REMOVED: volume_spike_threshold
        # REMOVED: rsi_confirm_period
        # REMOVED: max_volatility
        # REMOVED: momentum checks
        
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
        
        # Keep minimal other indicators for ranking
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Price_Change_5d'] = df['Close'].pct_change(5) * 100
        
        return df
    
    def check_quality_filters(self, symbol, date, df):
        """
        MINIMAL safety filters - Only block obvious garbage
        Returns (bool, str) - (passes, reason if failed)
        """
        try:
            if date not in df.index:
                return False, "Date not in data"
            
            row = df.loc[date]
            
            # Only block penny stocks
            if row['Close'] < self.min_price:
                return False, f"Penny stock: ${row['Close']:.2f}"
            
            # Only block if price is unrealistic
            if row['Close'] > self.max_price:
                return False, f"Price unrealistic: ${row['Close']:.2f}"
            
            # Only block extremely illiquid stocks
            if row['Volume'] < self.min_volume:
                return False, f"Extremely illiquid: {row['Volume']:,}"
            
            # That's it! Everything else passes
            return True, "Basic checks passed"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def run_backtest(self):
        """Execute the backtest"""
        print("\n🚀 Running V51.2 Minimal Filters Backtest...")
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
        print(f"🔍 Filter Mode: MINIMAL (only blocking obvious garbage)")
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
                        # Check minimal safety filters
                        passes_filter, reason = self.check_quality_filters(symbol, date, df)
                        
                        if passes_filter:
                            # Simple ranking by RSI (lower = better)
                            quality_score = self.rsi_entry - current_rsi
                            
                            entry_candidates.append({
                                'symbol': symbol,
                                'price': current_price,
                                'rsi': current_rsi,
                                'quality_score': quality_score
                            })
                            quality_accepted += 1
                        else:
                            quality_rejected += 1
                
                # Take the best signals (lowest RSI) up to our position limit
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
        print("V51.2 MINIMAL FILTERS RESULTS")
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
        print("FILTER IMPACT (MINIMAL MODE)")
        print("="*80)
        print(f"Signals Accepted:   {quality_accepted:>15,}")
        print(f"Signals Rejected:   {quality_rejected:>15,}")
        total_signals = quality_accepted + quality_rejected
        if total_signals > 0:
            rejection_rate = quality_rejected/total_signals*100
            print(f"Rejection Rate:     {rejection_rate:>15.1f}%")
            print(f"(Target: <20% for minimal filtering)")
        
        print("\n" + "="*80)
        print("💡 FINAL COMPARISON")
        print("="*80)
        print(f"V48 Original:       195.0% CAGR (NO filters, baseline)")
        print(f"V51 Original:        -5.3% CAGR (too strict, 96.9% rejection)")
        print(f"V51.1 Balanced:      12.0% CAGR (still strict, 69.2% rejection)")
        print(f"V51.2 Minimal:      {cagr:>5.1f}% CAGR (minimal filters, {rejection_rate:.1f}% rejection)")
        
        v48_diff = cagr - 195
        print(f"\n📊 Vs V48 Baseline: {v48_diff:+.1f}%")
        
        # Decision recommendation
        print("\n" + "="*80)
        print("🎯 RECOMMENDATION")
        print("="*80)
        
        if cagr >= 150:
            print("✅ SUCCESS! V51.2 is close to V48 performance.")
            print("   → Consider using V51.2 (minimal safety with good returns)")
        elif cagr >= 100:
            print("⚠️  MODERATE: V51.2 shows improvement but below V48.")
            print("   → Decision: Is the safety worth the performance cost?")
            print("   → If prioritize returns: Use V48")
            print("   → If prioritize safety: Use V51.2")
        else:
            print("❌ FILTERS NOT HELPING: Even minimal filters hurt performance.")
            print("   → RECOMMENDATION: Revert to V48")
            print("   → V48's 195% CAGR without filters is superior")
            print("   → The market is better at filtering than our rules")
        
        print(f"\n📊 Win Rate:")
        win_rate = len(winning) / len(trades_df) * 100
        print(f"V51.2 Win Rate:     {win_rate:.1f}%")
        if win_rate >= 50:
            print(f"✅ Good win rate")
        else:
            print(f"⚠️  Win rate below 50%")
        
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
        trades_df.to_csv('v51_minimal_trades.csv', index=False)
        equity_df.to_csv('v51_minimal_equity.csv', index=False)
        
        print(f"\n📁 Files saved:")
        print(f"  • v51_minimal_trades.csv ({len(trades_df)} trades)")
        print(f"  • v51_minimal_equity.csv ({len(equity_df)} data points)")
        
        print("\n" + "="*80)
        
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
    ║                   V51.2 MINIMAL FILTERS                               ║
    ║                                                                        ║
    ║  FINAL ATTEMPT - Almost no filtering, just basic safety              ║
    ║                                                                        ║
    ║  Active Filters:                                                      ║
    ║    • Only block penny stocks (< $2)                                   ║
    ║    • Only block extremely illiquid (< 100K volume)                    ║
    ║                                                                        ║
    ║  Removed:                                                             ║
    ║    ✗ Volume spike requirement                                         ║
    ║    ✗ RSI confirmation                                                 ║
    ║    ✗ Volatility limits                                                ║
    ║    ✗ Momentum checks                                                  ║
    ║                                                                        ║
    ║  Goal: Match V48's 195% CAGR while blocking true garbage             ║
    ║  If this doesn't work → Revert to V48                                ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    strategy = V51Minimal(initial_capital=30000)
    strategy.load_data(start_date='2024-01-01')
    results = strategy.run_backtest()
    
    print("\n✅ Backtest Complete!")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("Review the recommendation above.")
    print("If V51.2 doesn't significantly beat V51.1, revert to V48.")
    print("V48's simple approach (no filters) may be the winner.")
    print("="*80)
