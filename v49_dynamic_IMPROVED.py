#!/usr/bin/env python3
"""
V4.9 - WINNERS ONLY - DYNAMIC POSITION SIZING
Same as your original V48/V49 but with TRUE COMPOUNDING

KEY CHANGE: Position sizes now scale with total EQUITY, not just available cash
This enables exponential growth on winning periods

Original: 195% CAGR with static sizing
Dynamic: Expected 220-250%+ CAGR (let's find out!)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V49Dynamic:
    def __init__(self, initial_capital=30000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_equity = initial_capital  # NEW: Track total equity
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # 5 original high-beta symbols + extras
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        # V4 parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # Position management
        self.max_positions = 5  # UPDATED: 5 positions for better diversification
        self.position_size = 0.35  # 35% of EQUITY (not just cash!)
        self.leverage = 2.0
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def load_data(self, start_date='2025-01-01', end_date=None):
        """Load data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n📊 Loading V4.9 DYNAMIC data...")
        print(f"Period: {start_date} to {end_date}\n")
        self.data = {}
        
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                # Flatten multi-level columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) > 100:
                    # Calculate all indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    self.data[symbol] = df
                    print(f"✓ {symbol}: {len(df)} days")
            except Exception as e:
                print(f"✗ {symbol}: {e}")
        
        print(f"\n✓ Loaded {len(self.data)} symbols\n")
    
    def update_equity(self, date):
        """
        CRITICAL: Calculate total account equity
        This is the KEY difference from static version!
        """
        total_equity = self.cash
        
        for symbol, pos in self.positions.items():
            if date in self.data[symbol].index:
                try:
                    current = self.data[symbol].loc[date]
                    if isinstance(current, pd.DataFrame):
                        current = current.iloc[0]
                    current_price = float(current['Close'])
                    current_value = pos['shares'] * current_price
                    pnl = current_value - pos['leveraged_value']
                    total_equity += pos['entry_value'] + pnl
                except:
                    continue
        
        self.total_equity = total_equity
        return total_equity
    
    def generate_signals(self, date):
        """Generate signals using ONLY 2 winning strategies"""
        signals = []
        
        for symbol, df in self.data.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 50:
                continue
            
            try:
                current = df.loc[date]
                if isinstance(current, pd.DataFrame):
                    current = current.iloc[0]
                
                # Extract values safely
                rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                close = float(current['Close'])
                ma_fast = float(current['MA_Fast']) if not pd.isna(current['MA_Fast']) else 0
                ma_slow = float(current['MA_Slow']) if not pd.isna(current['MA_Slow']) else 0
                
                # Skip if missing data
                if rsi == 0 or ma_fast == 0 or ma_slow == 0:
                    continue
                
                strategy = None
                quality = 0
                
                # TREND_FOLLOW (Best performer from V4.8)
                if rsi > 60 and close > ma_fast > ma_slow:
                    strategy = 'TREND_FOLLOW'
                    quality = 3
                
                # PULLBACK (Second best from V4.8)  
                elif 45 <= rsi <= 55 and close > ma_slow and ma_fast > ma_slow:
                    strategy = 'PULLBACK'
                    quality = 2
                
                if strategy:
                    signals.append({
                        'symbol': symbol,
                        'date': date,
                        'price': close,
                        'strategy': strategy,
                        'quality': quality,
                        'rsi': rsi
                    })
                    
            except Exception as e:
                continue
        
        return signals
    
    def check_exits(self, date):
        """Check exit conditions"""
        exits = []
        
        for symbol, position in list(self.positions.items()):
            if date not in self.data[symbol].index:
                continue
                
            try:
                current = self.data[symbol].loc[date]
                if isinstance(current, pd.DataFrame):
                    current = current.iloc[0]
                
                current_price = float(current['Close'])
                current_rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                
                entry_price = position['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price)
                days_held = (date - position['entry_date']).days
                
                exit_reason = None
                
                # Stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'STOP_LOSS'
                
                # Take profit
                elif pnl_pct >= self.take_profit_pct:
                    exit_reason = 'TAKE_PROFIT'
                
                # RSI exit
                elif current_rsi > 0 and current_rsi < self.rsi_sell:
                    exit_reason = 'RSI_SELL'
                
                # Time exit
                elif days_held >= 14:
                    exit_reason = 'TIME_EXIT'
                
                if exit_reason:
                    exits.append({
                        'symbol': symbol,
                        'exit_price': current_price,
                        'date': date,
                        'exit_reason': exit_reason
                    })
                    
            except Exception as e:
                continue
        
        return exits
    
    def execute_trade(self, signal):
        """
        Execute buy trade - DYNAMIC SIZING VERSION
        Now uses TOTAL EQUITY, not just available cash!
        """
        if signal['symbol'] in self.positions:
            return
        
        if self.cash < 1000:
            return
        
        # CRITICAL CHANGE: Use total_equity instead of self.cash
        position_value = self.total_equity * self.position_size  # ← DYNAMIC!
        leveraged_value = position_value * self.leverage
        shares = leveraged_value / signal['price']
        
        # But still need enough cash to deploy
        cash_needed = position_value
        if cash_needed > self.cash:
            return
        
        # Record position
        self.positions[signal['symbol']] = {
            'entry_date': signal['date'],
            'entry_price': signal['price'],
            'shares': shares,
            'entry_value': position_value,
            'leveraged_value': leveraged_value,
            'strategy': signal['strategy'],
            'quality': signal['quality']
        }
        
        # Reduce cash
        self.cash -= position_value
        
        print(f"🟢 BUY {signal['symbol']}: {shares:.2f} shares @ ${signal['price']:.2f} "
              f"[{signal['strategy']}] Q:{signal['quality']}/3 "
              f"Size=${leveraged_value:,.0f} (Equity=${self.total_equity:,.0f})")
    
    def execute_exit(self, exit):
        """Execute sell trade"""
        symbol = exit['symbol']
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        exit_date = exit['date']
        exit_price = exit['exit_price']
        exit_reason = exit['exit_reason']
        
        # Calculate P&L
        shares = position['shares']
        entry_price = position['entry_price']
        leveraged_entry = position['leveraged_value']
        
        exit_value = shares * exit_price
        pnl = exit_value - leveraged_entry
        pnl_pct = (pnl / leveraged_entry) * 100
        
        # Return capital + profits
        self.cash += position['entry_value'] + pnl
        
        # Record trade
        self.trades.append({
            'entry_date': position['entry_date'],
            'exit_date': exit_date,
            'symbol': symbol,
            'strategy': position['strategy'],
            'quality': position['quality'],
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'leveraged_value': leveraged_entry,
            'exit_value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': (exit_date - position['entry_date']).days,
            'exit_reason': exit_reason
        })
        
        del self.positions[symbol]
        
        emoji = '🟢' if pnl > 0 else '🔴'
        print(f"{emoji} SELL {symbol}: ${pnl:,.0f} ({pnl_pct:+.1f}%) | {exit_reason} | "
              f"{(exit_date - position['entry_date']).days}d | Equity=${self.cash + sum(p['entry_value'] for p in self.positions.values()):,.0f}")
    
    def run_backtest(self):
        """Run backtest with DYNAMIC position sizing"""
        print("\n" + "="*80)
        print("V4.9 - WINNERS ONLY - DYNAMIC POSITION SIZING")
        print("Position sizes scale with TOTAL EQUITY (not just cash)")
        print("="*80)
        
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        print(f"\n📅 Period: {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"💰 Initial: ${self.initial_capital:,.0f}")
        print(f"📊 Strategy: Dynamic 2.0x | 35% per position | Max 5 positions")
        print(f"📊 Total Deployment: {5 * 0.35 * 2.0 * 100:.0f}% (350%)\n")
        
        signals_count = 0
        trades_count = 0
        
        for i, date in enumerate(all_dates):
            # Update equity FIRST (critical for dynamic sizing)
            total_equity = self.update_equity(date)
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Process exits
            for exit in self.check_exits(date):
                self.execute_exit(exit)
            
            # Update equity again after exits
            total_equity = self.update_equity(date)
            
            # Process new signals
            if len(self.positions) < self.max_positions:
                signals = self.generate_signals(date)
                signals_count += len(signals)
                
                # Sort by quality
                signals = sorted(signals, key=lambda x: x['quality'], reverse=True)
                
                for signal in signals[:self.max_positions - len(self.positions)]:
                    if signal['symbol'] not in self.positions:
                        self.execute_trade(signal)
                        trades_count += 1
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"Day {i+1}/{len(all_dates)} | Equity: ${total_equity:,.0f} | "
                      f"Cash: ${self.cash:,.0f} | Positions: {len(self.positions)}/5 | Trades: {trades_count}")
        
        # Close remaining
        if self.positions:
            final_date = all_dates[-1]
            print(f"\n🔴 Closing {len(self.positions)} positions:")
            for symbol in list(self.positions.keys()):
                if final_date in self.data[symbol].index:
                    try:
                        current = self.data[symbol].loc[final_date]
                        if isinstance(current, pd.DataFrame):
                            current = current.iloc[0]
                        exit_price = float(current['Close'])
                        self.execute_exit({
                            'symbol': symbol,
                            'exit_price': exit_price,
                            'date': final_date,
                            'exit_reason': 'BACKTEST_END'
                        })
                    except:
                        continue
        
        print(f"\n📊 Signals: {signals_count} | Trades: {trades_count}")
        self.generate_reports()
    
    def generate_reports(self):
        """Generate reports"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0:
            print("\n⚠️  NO TRADES EXECUTED")
            return
        
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
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS - V4.9 DYNAMIC POSITION SIZING")
        print("="*80)
        print(f"Initial Capital:    ${self.initial_capital:>15,.0f}")
        print(f"Final Equity:       ${final_equity:>15,.0f}")
        print(f"Total Return:       {total_return:>15.1f}%")
        print(f"CAGR:               {cagr:>15.1f}%")
        print(f"Max Drawdown:       {max_dd:>15.1f}%")
        print(f"\nTotal Trades:       {len(trades_df):>15,}")
        print(f"Winners:            {len(winning):>15,} ({len(winning)/len(trades_df)*100:.1f}%)")
        print(f"Losers:             {len(losing):>15,}")
        print(f"Avg Win:            ${winning['pnl'].mean():>14,.0f}")
        print(f"Avg Loss:           ${losing['pnl'].mean():>14,.0f}" if len(losing) > 0 else "Avg Loss:           $0")
        print(f"Avg Hold:           {trades_df['days_held'].mean():>15.1f} days")
        
        # Win/Loss Ratio
        if len(losing) > 0:
            win_loss_ratio = abs(winning['pnl'].mean() / losing['pnl'].mean())
            print(f"Win/Loss Ratio:     {win_loss_ratio:>15.2f}x")
        
        # Profit Factor
        if len(losing) > 0:
            profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum())
            print(f"Profit Factor:      {profit_factor:>15.2f}")
        
        print("="*80)
        
        # Strategy breakdown
        print("\nSTRATEGY BREAKDOWN:")
        strategy_stats = trades_df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(0)
        print(strategy_stats)
        
        # Symbol breakdown
        print("\nSYMBOL BREAKDOWN:")
        symbol_stats = trades_df.groupby('symbol').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(0)
        print(symbol_stats)
        
        # Save files
        trades_df.to_csv('v49_dynamic_trades.csv', index=False)
        equity_df.to_csv('v49_dynamic_equity.csv', index=False)
        
        print(f"\n📁 Files saved:")
        print(f"   • v49_dynamic_trades.csv - All {len(trades_df)} trades")
        print(f"   • v49_dynamic_equity.csv")
        
        print("\n" + "="*80)
        print("💡 DYNAMIC vs STATIC COMPARISON")
        print("="*80)
        print("Your STATIC version: 195% CAGR (2025 YTD)")
        print(f"This DYNAMIC version: {cagr:.1f}% CAGR (2025 YTD)")
        print(f"Improvement: {cagr - 195:.1f}% CAGR")
        print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("V4.9 - DYNAMIC POSITION SIZING TEST")
    print("Same strategy as your 195% CAGR version")
    print("But with TRUE COMPOUNDING on total equity")
    print("="*80)
    
    strategy = V49Dynamic(initial_capital=30000)
    strategy.load_data(start_date='2025-01-01')
    strategy.run_backtest()
    
    print("\n✅ Complete!")
    print("\nCompare:")
    print("  • v49_dynamic_trades.csv (this run)")
    print("  • v49_winners_only_trades.csv (your original)")
