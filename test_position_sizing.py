#!/usr/bin/env python3
"""
V4.9 Position Sizing Comparison Test
Compare 3 positions @ 35% vs 5 positions @ 25%
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V49PositionTest:
    def __init__(self, initial_capital=30000, max_positions=3, position_size=0.35):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size = position_size
        
        # Symbols including NET and META
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        # V4.9 Parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        self.leverage = 2.0
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        self.max_hold_days = 14
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def load_data(self, start_date='2025-01-01'):
        """Load data for all symbols"""
        print(f"📊 Loading data for {len(self.symbols)} symbols...")
        self.data = {}
        
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=start_date, progress=False, auto_adjust=True)
                if len(df) > 100:
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    df = df.dropna()
                    self.data[symbol] = df
                    print(f"  ✓ {symbol}: {len(df)} days")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        print(f"✅ Loaded {len(self.data)} symbols\n")
    
    def check_trend_follow(self, row, prev_row):
        """TREND_FOLLOW strategy"""
        try:
            rsi = float(row['RSI'])
            close = float(row['Close'])
            ma_fast = float(row['MA_Fast'])
            ma_slow = float(row['MA_Slow'])
            return (rsi > 55 and close > ma_fast and close > ma_slow and ma_fast > ma_slow)
        except:
            return False
    
    def check_pullback(self, row, prev_row):
        """PULLBACK strategy"""
        try:
            rsi = float(row['RSI'])
            close = float(row['Close'])
            ma_fast = float(row['MA_Fast'])
            ma_slow = float(row['MA_Slow'])
            prev_rsi = float(prev_row['RSI'])
            return (45 <= rsi <= 55 and close > ma_slow and ma_fast > ma_slow and prev_rsi > 55)
        except:
            return False
    
    def run_backtest(self):
        """Run backtest with current position settings"""
        
        # Get all unique dates
        all_dates = sorted(set().union(*[set(df.index) for df in self.data.values()]))
        
        capital = self.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        
        for date in all_dates:
            # Check exits
            for symbol in list(positions.keys()):
                if symbol not in self.data or date not in self.data[symbol].index:
                    continue
                
                pos = positions[symbol]
                current_price = float(self.data[symbol].loc[date, 'Close'])
                entry_price = pos['entry_price']
                days_held = (date - pos['entry_date']).days
                pnl_pct = (current_price - entry_price) / entry_price
                rsi = float(self.data[symbol].loc[date, 'RSI'])
                
                exit_signal = False
                exit_reason = None
                
                if pnl_pct <= -self.stop_loss_pct:
                    exit_signal = True
                    exit_reason = 'STOP_LOSS'
                elif pnl_pct >= self.take_profit_pct:
                    exit_signal = True
                    exit_reason = 'TAKE_PROFIT'
                elif rsi < self.rsi_sell:
                    exit_signal = True
                    exit_reason = 'RSI_SELL'
                elif days_held >= self.max_hold_days:
                    exit_signal = True
                    exit_reason = 'TIME_EXIT'
                
                if exit_signal:
                    shares = pos['shares']
                    exit_value = shares * current_price
                    pnl = exit_value - pos['entry_value']
                    capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'days_held': days_held,
                        'exit_reason': exit_reason
                    })
                    
                    del positions[symbol]
            
            # Check entries (if we have room)
            if len(positions) < self.max_positions:
                signals = []
                
                for symbol in self.symbols:
                    if symbol in positions or symbol not in self.data:
                        continue
                    
                    if date not in self.data[symbol].index:
                        continue
                    
                    df = self.data[symbol]
                    idx = df.index.get_loc(date)
                    
                    if idx < 1:
                        continue
                    
                    current = df.iloc[idx]
                    previous = df.iloc[idx-1]
                    
                    signal = False
                    strategy = None
                    
                    if self.check_trend_follow(current, previous):
                        signal = True
                        strategy = 'TREND_FOLLOW'
                    elif self.check_pullback(current, previous):
                        signal = True
                        strategy = 'PULLBACK'
                    
                    if signal:
                        signals.append({
                            'symbol': symbol,
                            'strategy': strategy,
                            'price': float(current['Close']),
                            'rsi': float(current['RSI'])
                        })
                
                # Take top signals up to max positions
                slots_available = self.max_positions - len(positions)
                for signal in signals[:slots_available]:
                    position_value = capital * self.position_size * self.leverage
                    shares = int(position_value / signal['price'])
                    
                    if shares > 0:
                        positions[signal['symbol']] = {
                            'entry_date': date,
                            'entry_price': signal['price'],
                            'entry_value': shares * signal['price'],
                            'shares': shares,
                            'strategy': signal['strategy']
                        }
            
            # Track equity
            position_value = sum(
                pos['shares'] * float(self.data[sym].loc[date, 'Close'])
                for sym, pos in positions.items()
                if sym in self.data and date in self.data[sym].index
            )
            total_equity = capital + position_value - self.initial_capital
            equity_curve.append({
                'date': date,
                'equity': capital + position_value,
                'positions': len(positions)
            })
        
        # Close remaining positions
        final_date = all_dates[-1]
        for symbol, pos in positions.items():
            if symbol in self.data and final_date in self.data[symbol].index:
                final_price = float(self.data[symbol].loc[final_date, 'Close'])
                shares = pos['shares']
                pnl = (shares * final_price) - pos['entry_value']
                capital += pnl
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': final_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': ((final_price - pos['entry_price']) / pos['entry_price']) * 100,
                    'days_held': (final_date - pos['entry_date']).days,
                    'exit_reason': 'BACKTEST_END'
                })
        
        return trades, equity_curve, capital
    
    def analyze_results(self, trades, equity_curve, final_capital, label):
        """Analyze and print results"""
        
        if len(trades) == 0:
            print(f"{label}: No trades generated")
            return None
        
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        years = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25
        cagr = (((final_capital / self.initial_capital) ** (1 / years)) - 1) * 100
        
        wins = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (wins / len(trades_df)) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if wins < len(trades_df) else 0
        
        # Calculate max drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Average positions held
        avg_positions = equity_df['positions'].mean()
        
        return {
            'label': label,
            'max_positions': self.max_positions,
            'position_size': self.position_size,
            'total_trades': len(trades_df),
            'final_capital': final_capital,
            'total_return': total_return,
            'cagr': cagr,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'avg_positions': avg_positions,
            'avg_hold': trades_df['days_held'].mean()
        }

def run_comparison():
    """Run comparison test"""
    print("\n" + "="*80)
    print("V4.9 POSITION SIZING COMPARISON TEST - 2025 YTD")
    print("Start Date: 2025-01-01 | Symbols: 7 (NVDA, TSLA, PLTR, AMD, COIN, META, NET)")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: Current setup (3 positions @ 35%)
    print("Testing: 3 positions @ 35% each...")
    test1 = V49PositionTest(max_positions=3, position_size=0.35)
    test1.load_data()
    trades1, equity1, capital1 = test1.run_backtest()
    result1 = test1.analyze_results(trades1, equity1, capital1, "3 Pos @ 35%")
    results.append(result1)
    print(f"✅ Complete: {result1['total_trades']} trades, {result1['cagr']:.1f}% CAGR\n")
    
    # Test 2: 5 positions @ 25%
    print("Testing: 5 positions @ 25% each...")
    test2 = V49PositionTest(max_positions=5, position_size=0.25)
    test2.load_data()
    trades2, equity2, capital2 = test2.run_backtest()
    result2 = test2.analyze_results(trades2, equity2, capital2, "5 Pos @ 25%")
    results.append(result2)
    print(f"✅ Complete: {result2['total_trades']} trades, {result2['cagr']:.1f}% CAGR\n")
    
    # Test 3: 5 positions @ 21% (same total leverage as 3@35%)
    print("Testing: 5 positions @ 21% each (same total leverage)...")
    test3 = V49PositionTest(max_positions=5, position_size=0.21)
    test3.load_data()
    trades3, equity3, capital3 = test3.run_backtest()
    result3 = test3.analyze_results(trades3, equity3, capital3, "5 Pos @ 21%")
    results.append(result3)
    print(f"✅ Complete: {result3['total_trades']} trades, {result3['cagr']:.1f}% CAGR\n")
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")
    
    print(f"{'Metric':<25} {'3 Pos @ 35%':<20} {'5 Pos @ 25%':<20} {'5 Pos @ 21%':<20}")
    print("-" * 85)
    
    metrics = [
        ('Total Leverage', lambda r: f"{r['max_positions'] * r['position_size'] * 2:.0f}%", ''),
        ('Total Trades', lambda r: f"{r['total_trades']}", ''),
        ('Avg Positions Held', lambda r: f"{r['avg_positions']:.1f}", ''),
        ('Final Capital', lambda r: f"${r['final_capital']:,.0f}", ''),
        ('Total Return', lambda r: f"{r['total_return']:.1f}%", ''),
        ('CAGR', lambda r: f"{r['cagr']:.1f}%", '%'),
        ('Win Rate', lambda r: f"{r['win_rate']:.1f}%", '%'),
        ('Avg Win', lambda r: f"${r['avg_win']:,.0f}", ''),
        ('Avg Loss', lambda r: f"${r['avg_loss']:,.0f}", ''),
        ('Max Drawdown', lambda r: f"{r['max_drawdown']:.1f}%", '%'),
        ('Avg Hold Days', lambda r: f"{r['avg_hold']:.1f}", 'd'),
    ]
    
    for metric_name, metric_func, suffix in metrics:
        values = [metric_func(r) for r in results]
        print(f"{metric_name:<25} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    # Determine winner
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80 + "\n")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['cagr'])
    best = results[best_idx]
    
    print(f"🏆 WINNER: {best['label']}")
    print(f"   CAGR: {best['cagr']:.1f}%")
    print(f"   Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"   Total Trades: {best['total_trades']}\n")
    
    # Analysis
    cagr_diff_5_25 = results[1]['cagr'] - results[0]['cagr']
    cagr_diff_5_21 = results[2]['cagr'] - results[0]['cagr']
    
    print("💡 ANALYSIS:\n")
    
    print(f"Moving from 3 positions @ 35% to 5 positions @ 25%:")
    print(f"  CAGR change: {cagr_diff_5_25:+.1f} percentage points")
    print(f"  Trade count: {results[1]['total_trades'] - results[0]['total_trades']:+d} trades")
    print(f"  Drawdown: {results[1]['max_drawdown'] - results[0]['max_drawdown']:+.1f}%\n")
    
    print(f"Moving from 3 positions @ 35% to 5 positions @ 21%:")
    print(f"  CAGR change: {cagr_diff_5_21:+.1f} percentage points")
    print(f"  Trade count: {results[2]['total_trades'] - results[0]['total_trades']:+d} trades")
    print(f"  Drawdown: {results[2]['max_drawdown'] - results[0]['max_drawdown']:+.1f}%\n")
    
    if abs(cagr_diff_5_25) < 5:
        print("⚠️  Difference is marginal (<5%). Stick with simpler 3-position setup.")
    elif cagr_diff_5_25 > 10:
        print("✅ Significant improvement! Consider switching to 5 positions.")
    else:
        print("📊 Moderate improvement. Consider complexity vs. reward trade-off.")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_comparison()