#!/usr/bin/env python3
"""
QBTS Backtest with V4.8 and V4.9 Momentum Strategies
Tests whether QBTS would have been profitable with your momentum approach
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class QBTSBacktest:
    def __init__(self, initial_capital=7000):  # 35% of $20k position
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = None
        self.trades = []
        
        # V4 Strategy Parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        self.leverage = 2.0
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_trend_follow(self, row, prev_row):
        """Strategy 1: TREND_FOLLOW - Strong uptrend momentum"""
        if pd.isna(prev_row['RSI']):
            return False
        return (row['RSI'] > 55 and 
                row['Close'] > row['MA_Fast'] and 
                row['Close'] > row['MA_Slow'] and
                row['MA_Fast'] > row['MA_Slow'])
    
    def check_pullback(self, row, prev_row):
        """Strategy 2: PULLBACK - Dip in uptrend"""
        if pd.isna(prev_row['RSI']):
            return False
        return (45 <= row['RSI'] <= 55 and 
                row['Close'] > row['MA_Slow'] and
                row['MA_Fast'] > row['MA_Slow'] and
                prev_row['RSI'] > 55)
    
    def check_breakout(self, row, df, idx):
        """Strategy 3: BREAKOUT - New highs with volume"""
        if idx < 20:
            return False
        lookback = df.iloc[max(0, idx-20):idx]
        if len(lookback) == 0:
            return False
        return (row['Close'] > lookback['High'].max() and 
                row['Volume'] > row['Volume_MA'] * 1.5)
    
    def check_momentum_accel(self, row, df, idx):
        """Strategy 4: MOMENTUM_ACCEL - RSI rising fast"""
        if idx < 3:
            return False
        rsi_values = df.iloc[idx-3:idx+1]['RSI'].values
        if len(rsi_values) < 4:
            return False
        return (all(rsi_values[i] < rsi_values[i+1] for i in range(3)) and 
                row['Volume'] > row['Volume_MA'] * 1.2)
    
    def check_oversold_bounce(self, row, prev_row):
        """Strategy 5: OVERSOLD_BOUNCE - Recovery from oversold"""
        if pd.isna(prev_row['RSI']):
            return False
        return (prev_row['RSI'] < 30 and 
                row['RSI'] > 30 and 
                row['Close'] > row['MA_Fast'])
    
    def check_higher_low(self, row, df, idx):
        """Strategy 6: HIGHER_LOW - Making higher lows"""
        if idx < 10:
            return False
        lookback = df.iloc[max(0, idx-10):idx]
        if len(lookback) < 2:
            return False
        lows = lookback['Low'].values
        return (len(lows) >= 2 and 
                lows[-1] > lows[-2] and 
                row['RSI'] > 55)
    
    def run_backtest(self, strategies='all', start_date='2021-01-01'):
        """Run backtest with specified strategies"""
        
        # Download QBTS data
        print(f"\n{'='*80}")
        print(f"QBTS BACKTEST - {strategies.upper()} STRATEGIES")
        print(f"{'='*80}")
        print(f"\n📊 Downloading QBTS data from {start_date}...\n")
        
        df = yf.download('QBTS', start=start_date, progress=False)
        
        if len(df) == 0:
            print("❌ No data available for QBTS")
            return
        
        # Calculate indicators
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
        df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df = df.dropna()
        
        print(f"✓ Loaded {len(df)} trading days")
        print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"  Current price: ${df['Close'].iloc[-1]:.2f}\n")
        
        # Reset for backtest
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        
        # Run through each day
        for idx in range(1, len(df)):
            date = df.index[idx]
            row = df.iloc[idx]
            prev_row = df.iloc[idx-1]
            
            # Check for exit signals first
            if self.position:
                entry_price = self.position['entry_price']
                current_price = row['Close']
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Exit conditions
                exit_signal = False
                exit_reason = None
                
                if pnl_pct <= -self.stop_loss_pct:
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
                elif pnl_pct >= self.take_profit_pct:
                    exit_signal = True
                    exit_reason = 'TAKE_PROFIT'
                elif row['RSI'] < self.rsi_sell:
                    exit_signal = True
                    exit_reason = 'RSI_EXIT'
                
                if exit_signal:
                    # Close position
                    shares = self.position['shares']
                    exit_value = shares * current_price
                    pnl = exit_value - self.position['entry_value']
                    
                    self.capital += pnl
                    
                    self.trades.append({
                        'entry_date': self.position['entry_date'],
                        'exit_date': date,
                        'strategy': self.position['strategy'],
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': exit_reason,
                        'capital_after': self.capital
                    })
                    
                    self.position = None
            
            # Check for entry signals if no position
            if not self.position:
                signal = False
                strategy_name = None
                
                # Check strategies based on mode
                if strategies == 'all':
                    # V4.8: All 6 strategies
                    if self.check_trend_follow(row, prev_row):
                        signal = True
                        strategy_name = 'TREND_FOLLOW'
                    elif self.check_pullback(row, prev_row):
                        signal = True
                        strategy_name = 'PULLBACK'
                    elif self.check_breakout(row, df, idx):
                        signal = True
                        strategy_name = 'BREAKOUT'
                    elif self.check_momentum_accel(row, df, idx):
                        signal = True
                        strategy_name = 'MOMENTUM_ACCEL'
                    elif self.check_oversold_bounce(row, prev_row):
                        signal = True
                        strategy_name = 'OVERSOLD_BOUNCE'
                    elif self.check_higher_low(row, df, idx):
                        signal = True
                        strategy_name = 'HIGHER_LOW'
                else:
                    # V4.9: Winners only
                    if self.check_trend_follow(row, prev_row):
                        signal = True
                        strategy_name = 'TREND_FOLLOW'
                    elif self.check_pullback(row, prev_row):
                        signal = True
                        strategy_name = 'PULLBACK'
                
                if signal:
                    # Enter position with leverage
                    position_value = self.capital * self.leverage
                    shares = int(position_value / row['Close'])
                    
                    if shares > 0:
                        self.position = {
                            'entry_date': date,
                            'entry_price': row['Close'],
                            'entry_value': shares * row['Close'],
                            'shares': shares,
                            'strategy': strategy_name
                        }
        
        # Close any open position at end
        if self.position:
            final_price = df['Close'].iloc[-1]
            shares = self.position['shares']
            exit_value = shares * final_price
            pnl = exit_value - self.position['entry_value']
            pnl_pct = (final_price - self.position['entry_price']) / self.position['entry_price']
            
            self.capital += pnl
            
            self.trades.append({
                'entry_date': self.position['entry_date'],
                'exit_date': df.index[-1],
                'strategy': self.position['strategy'],
                'entry_price': self.position['entry_price'],
                'exit_price': final_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'exit_reason': 'END_OF_PERIOD',
                'capital_after': self.capital
            })
        
        # Print results
        self.print_results(strategies)
    
    def print_results(self, strategy_type):
        """Print backtest results"""
        
        print(f"\n{'='*80}")
        print(f"RESULTS - {strategy_type.upper()} STRATEGIES")
        print(f"{'='*80}\n")
        
        if len(self.trades) == 0:
            print("❌ No trades generated")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        total_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = (wins / len(trades_df)) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0
        
        print(f"💰 PERFORMANCE:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital:   ${self.capital:,.2f}")
        print(f"  Total Return:    {total_return:+.1f}%")
        print(f"  Total P&L:       ${self.capital - self.initial_capital:+,.2f}\n")
        
        print(f"📊 TRADE STATISTICS:")
        print(f"  Total Trades:    {len(trades_df)}")
        print(f"  Wins:           {wins} ({win_rate:.1f}%)")
        print(f"  Losses:         {losses} ({100-win_rate:.1f}%)")
        print(f"  Avg Win:        ${avg_win:+,.2f}")
        print(f"  Avg Loss:       ${avg_loss:+,.2f}")
        if losses > 0:
            print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}x\n")
        
        # Strategy breakdown
        print(f"📈 BY STRATEGY:")
        strategy_stats = trades_df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean'
        }).round(2)
        
        for strategy in strategy_stats.index:
            strat_trades = trades_df[trades_df['strategy'] == strategy]
            strat_wins = len(strat_trades[strat_trades['pnl'] > 0])
            strat_total = len(strat_trades)
            strat_win_rate = (strat_wins / strat_total * 100) if strat_total > 0 else 0
            
            print(f"  {strategy:20} {strat_total:3d} trades | "
                  f"${strategy_stats.loc[strategy, ('pnl', 'sum')]:+8,.0f} | "
                  f"{strat_win_rate:5.1f}% win rate | "
                  f"Avg: ${strategy_stats.loc[strategy, ('pnl', 'mean')]:+7,.0f}")
        
        # Recent trades
        print(f"\n📝 LAST 10 TRADES:")
        print(f"{'Date':<12} {'Strategy':<16} {'Entry':<8} {'Exit':<8} {'P&L':<12} {'%':<8} {'Reason':<15}")
        print(f"{'-'*95}")
        
        for _, trade in trades_df.tail(10).iterrows():
            print(f"{trade['exit_date'].strftime('%Y-%m-%d'):<12} "
                  f"{trade['strategy']:<16} "
                  f"${trade['entry_price']:>6.2f}  "
                  f"${trade['exit_price']:>6.2f}  "
                  f"${trade['pnl']:>+9.2f}  "
                  f"{trade['pnl_pct']:>+6.1f}%  "
                  f"{trade['exit_reason']:<15}")

# Run both versions
if __name__ == "__main__":
    print("\n" + "="*80)
    print("QBTS MOMENTUM STRATEGY BACKTEST")
    print("Testing if QBTS fits your V4.8 / V4.9 momentum strategy")
    print("="*80)
    
    # Test V4.8 (all 6 strategies)
    bt_v48 = QBTSBacktest()
    bt_v48.run_backtest(strategies='all', start_date='2021-01-01')
    
    print("\n\n")
    
    # Test V4.9 (winners only)
    bt_v49 = QBTSBacktest()
    bt_v49.run_backtest(strategies='winners', start_date='2021-01-01')
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if len(bt_v48.trades) > 0 and len(bt_v49.trades) > 0:
        v48_return = ((bt_v48.capital - bt_v48.initial_capital) / bt_v48.initial_capital) * 100
        v49_return = ((bt_v49.capital - bt_v49.initial_capital) / bt_v49.initial_capital) * 100
        
        if v48_return > 0 and v49_return > 0:
            print("\n✅ QBTS shows positive returns with both strategies")
            print(f"   V4.8 Return: {v48_return:+.1f}%")
            print(f"   V4.9 Return: {v49_return:+.1f}%")
            print("\n⚠️  However, consider:")
            print("   - QBTS is HIGHLY speculative")
            print("   - Recent price action may not repeat")
            print("   - Limited trading history compared to NVDA/TSLA")
            print("\n💡 Suggestion: Paper trade QBTS for 1-2 months before adding real capital")
        else:
            print("\n❌ QBTS shows negative returns with your strategy")
            print(f"   V4.8 Return: {v48_return:+.1f}%")
            print(f"   V4.9 Return: {v49_return:+.1f}%")
            print("\n💡 Recommendation: DO NOT add QBTS to your strategy")
    else:
        print("\n⚠️  Insufficient trades to make a determination")
        print("   QBTS may not trigger enough momentum signals")
    
    print("\n" + "="*80 + "\n")