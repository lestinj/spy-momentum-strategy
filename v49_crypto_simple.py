#!/usr/bin/env python3
"""
V4.9 CRYPTO EDITION - SIMPLIFIED
=================================

Fixed version that properly handles yfinance crypto data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V49Crypto:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Crypto universe
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 
                       'LINK-USD', 'MATIC-USD', 'DOT-USD']
        
        # Strategy parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # Position management
        self.max_positions = 3
        self.position_size = 0.35
        self.leverage = 1.5
        self.stop_loss_pct = 0.10
        self.take_profit_pct = 0.30
        self.max_hold_days = 14
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def load_data(self, start_date='2020-01-01', end_date=None):
        """Load crypto data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print("V4.9 CRYPTO - Loading Data")
        print(f"{'='*80}\n")
        
        self.data = {}
        
        for symbol in self.symbols:
            try:
                print(f"Downloading {symbol}...", end=' ')
                df = yf.download(symbol, start=start_date, end=end_date, 
                                progress=False, auto_adjust=True)
                
                # Ensure proper column structure
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if len(df) > 100:
                    # Calculate indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    # Drop NaN rows
                    df = df.dropna()
                    
                    self.data[symbol] = df
                    print(f"✓ {len(df)} days")
                else:
                    print(f"✗ Insufficient data")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"\n✓ Loaded {len(self.data)} crypto assets\n")
        return len(self.data) > 0
    
    def generate_signals(self, date):
        """Generate trading signals"""
        signals = []
        
        for symbol, df in self.data.items():
            if date not in df.index:
                continue
            
            try:
                row = df.loc[date]
                
                # Get values
                rsi = row['RSI']
                close = row['Close']
                ma_fast = row['MA_Fast']
                ma_slow = row['MA_Slow']
                
                # Skip if any NaN
                if pd.isna(rsi) or pd.isna(ma_fast) or pd.isna(ma_slow):
                    continue
                
                strategy = None
                
                # TREND_FOLLOW
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    strategy = 'TREND_FOLLOW'
                
                # PULLBACK
                elif (self.rsi_sell < rsi < self.rsi_buy and
                      close > ma_slow and
                      ma_fast > ma_slow):
                    strategy = 'PULLBACK'
                
                if strategy:
                    signals.append({
                        'symbol': symbol,
                        'price': float(close),
                        'date': date,
                        'strategy': strategy,
                        'rsi': float(rsi)
                    })
                    
            except Exception as e:
                continue
        
        return signals
    
    def check_exits(self, date):
        """Check exit conditions"""
        exits = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in self.data:
                continue
            if date not in self.data[symbol].index:
                continue
            
            try:
                row = self.data[symbol].loc[date]
                current_price = float(row['Close'])
                current_rsi = float(row['RSI'])
                
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                days_held = (date - position['entry_date']).days
                
                exit_reason = None
                
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'STOP_LOSS'
                elif pnl_pct >= self.take_profit_pct:
                    exit_reason = 'TAKE_PROFIT'
                elif current_rsi < self.rsi_sell:
                    exit_reason = 'RSI_SELL'
                elif days_held >= self.max_hold_days:
                    exit_reason = 'TIME_EXIT'
                
                if exit_reason:
                    exits.append({
                        'symbol': symbol,
                        'exit_price': current_price,
                        'date': date,
                        'exit_reason': exit_reason
                    })
                    
            except:
                continue
        
        return exits
    
    def execute_trade(self, signal):
        """Execute buy"""
        if signal['symbol'] in self.positions:
            return
        if self.capital < 1000:
            return
        
        position_value = self.capital * self.position_size
        leveraged_value = position_value * self.leverage
        shares = leveraged_value / signal['price']
        
        self.positions[signal['symbol']] = {
            'entry_price': signal['price'],
            'entry_date': signal['date'],
            'shares': shares,
            'entry_value': position_value,
            'leveraged_value': leveraged_value,
            'strategy': signal['strategy']
        }
        
        self.capital -= position_value
        
        clean = signal['symbol'].replace('-USD', '')
        print(f"🟢 BUY  {clean:6} ${signal['price']:>8,.0f} | {signal['strategy']}")
    
    def execute_exit(self, exit):
        """Execute sell"""
        symbol = exit['symbol']
        position = self.positions[symbol]
        
        current_value = position['shares'] * exit['exit_price']
        pnl = current_value - position['leveraged_value']
        pnl_pct = (exit['exit_price'] - position['entry_price']) / position['entry_price'] * 100
        
        self.capital += position['entry_value'] + pnl
        
        self.trades.append({
            'symbol': symbol,
            'strategy': position['strategy'],
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': exit['date'],
            'exit_price': exit['exit_price'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': (exit['date'] - position['entry_date']).days,
            'exit_reason': exit['exit_reason']
        })
        
        del self.positions[symbol]
        
        emoji = '🟢' if pnl > 0 else '🔴'
        clean = symbol.replace('-USD', '')
        print(f"{emoji} SELL {clean:6} ${exit['exit_price']:>8,.0f} | {pnl_pct:+6.1f}% | {exit['exit_reason']}")
    
    def run_backtest(self):
        """Run backtest"""
        print("="*80)
        print("V4.9 CRYPTO BACKTEST")
        print("="*80)
        
        # Get all dates
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        print(f"\n📅 {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"💰 Initial: ${self.initial_capital:,.0f}")
        print(f"🎯 {self.max_positions} positions @ {self.position_size*100:.0f}% @ {self.leverage}x\n")
        
        for i, date in enumerate(all_dates):
            # Calculate equity
            total_equity = self.capital
            for symbol, pos in self.positions.items():
                if symbol in self.data and date in self.data[symbol].index:
                    try:
                        price = float(self.data[symbol].loc[date]['Close'])
                        value = pos['shares'] * price
                        pnl = value - pos['leveraged_value']
                        total_equity += pos['entry_value'] + pnl
                    except:
                        pass
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity
            })
            
            # Exits
            for exit in self.check_exits(date):
                self.execute_exit(exit)
            
            # New entries
            if len(self.positions) < self.max_positions:
                signals = self.generate_signals(date)
                for signal in signals[:self.max_positions - len(self.positions)]:
                    self.execute_trade(signal)
            
            # Progress
            if (i + 1) % 365 == 0:
                print(f"Year {(i+1)/365:.1f} | ${total_equity:,.0f} | {len(self.trades)} trades")
        
        # Close remaining
        if self.positions:
            final_date = all_dates[-1]
            for symbol in list(self.positions.keys()):
                if symbol in self.data and final_date in self.data[symbol].index:
                    try:
                        price = float(self.data[symbol].loc[final_date]['Close'])
                        self.execute_exit({
                            'symbol': symbol,
                            'exit_price': price,
                            'date': final_date,
                            'exit_reason': 'END'
                        })
                    except:
                        pass
        
        self.show_results()
    
    def show_results(self):
        """Show results"""
        if len(self.trades) == 0:
            print("\n⚠️  NO TRADES\n")
            return
        
        df_equity = pd.DataFrame(self.equity_curve)
        df_trades = pd.DataFrame(self.trades)
        
        final = df_equity['equity'].iloc[-1]
        total_return = (final - self.initial_capital) / self.initial_capital * 100
        days = (df_equity['date'].iloc[-1] - df_equity['date'].iloc[0]).days
        cagr = ((final / self.initial_capital) ** (365.25 / days) - 1) * 100
        
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['dd'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak'] * 100
        max_dd = df_equity['dd'].min()
        
        winners = df_trades[df_trades['pnl'] > 0]
        losers = df_trades[df_trades['pnl'] <= 0]
        
        print(f"\n{'='*80}")
        print("CRYPTO RESULTS")
        print(f"{'='*80}")
        print(f"Final:        ${final:>12,.0f}")
        print(f"Return:       {total_return:>12.1f}%")
        print(f"CAGR:         {cagr:>12.1f}%")
        print(f"Max DD:       {max_dd:>12.1f}%")
        print(f"\nTrades:       {len(df_trades):>12,}")
        print(f"Win Rate:     {len(winners)/len(df_trades)*100:>12.1f}%")
        print(f"Avg Win:      ${winners['pnl'].mean():>11,.0f}")
        print(f"Avg Loss:     ${losers['pnl'].mean():>11,.0f}")
        
        print(f"\n{'='*80}")
        print("VS STOCKS")
        print(f"{'='*80}")
        print(f"Stocks:  $971K | 120.4% CAGR | -67.8% DD")
        print(f"Crypto:  ${final/1000:.0f}K | {cagr:.1f}% CAGR | {max_dd:.1f}% DD")
        
        if cagr > 120:
            print(f"\n🚀 CRYPTO WINS by {cagr-120.4:.1f}%!")
        else:
            print(f"\n⚠️  Stocks outperformed by {120.4-cagr:.1f}%")
        
        # Save
        df_trades.to_csv('v49_crypto_trades.csv', index=False)
        df_equity.to_csv('v49_crypto_equity.csv', index=False)
        print(f"\n📁 Saved: v49_crypto_trades.csv, v49_crypto_equity.csv")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    strategy = V49Crypto(initial_capital=10000)
    
    if strategy.load_data(start_date='2020-01-01'):
        strategy.run_backtest()
    else:
        print("❌ Failed to load data")
