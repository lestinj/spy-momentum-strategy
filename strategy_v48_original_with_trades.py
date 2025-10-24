#!/usr/bin/env python3
"""
V4.9 - WINNERS ONLY
Only uses the 2 profitable strategies from V4.8:
- TREND_FOLLOW: +$837K (227 trades, $3,686 avg)
- PULLBACK: +$344K (66 trades, $5,205 avg)

Removing losers: BREAKOUT, HIGHER_LOW, MOMENTUM_ACCEL, OVERSOLD_BOUNCE
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class V49WinnersOnly:
    def __init__(self, initial_capital=300000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # 5 original high-beta symbols
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        # V4 parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # Position management
        self.max_positions = 3
        self.position_size = 0.45  # 35%
        self.leverage = 2.5
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
    
    def load_data(self, start_date='24-01-01', end_date=None):
        """Load data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"\n📊 Loading V4.9 data with 2 WINNING strategies...")
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
                
                # Skip if indicators not ready
                if rsi == 0 or ma_fast == 0 or ma_slow == 0:
                    continue
                
                strategy_triggered = None
                quality = 1
                
                # STRATEGY 1: Trend Following (WINNER - $837K)
                # Buy when RSI > 55, price above both MAs, MAs aligned
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    strategy_triggered = 'TREND_FOLLOW'
                    quality = 3  # Upgraded from 2 to 3 since it's proven
                
                # STRATEGY 2: Pullback Entry (WINNER - $344K)
                # Buy when pullback in uptrend (RSI < 55 but price above MAs)
                elif (rsi < self.rsi_buy and rsi > 45 and
                      close > ma_slow and
                      ma_fast > ma_slow):
                    strategy_triggered = 'PULLBACK'
                    quality = 3  # Upgraded from 2 to 3 since it's proven
                
                if strategy_triggered:
                    signals.append({
                        'symbol': symbol,
                        'price': close,
                        'date': date,
                        'strategy': strategy_triggered,
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
        """Execute buy trade"""
        if signal['symbol'] in self.positions:
            return
        
        if self.capital < 1000:
            return
        
        # Calculate position size
        position_value = self.capital * self.position_size
        leveraged_value = position_value * self.leverage
        shares = leveraged_value / signal['price']
        
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
        
        # Reduce capital
        self.capital -= position_value
        
        print(f"🟢 BUY {signal['symbol']}: {shares:.2f} shares @ ${signal['price']:.2f} [{signal['strategy']}] Quality:{signal['quality']}/3 RSI:{signal['rsi']:.0f}")
    
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
        self.capital += position['entry_value'] + pnl
        
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
        print(f"{emoji} SELL {symbol}: ${pnl:,.0f} ({pnl_pct:+.1f}%) | {exit_reason} | {(exit_date - position['entry_date']).days}d")
    
    def run_backtest(self):
        """Run backtest"""
        print("\n" + "="*80)
        print("V4.9 - WINNERS ONLY")
        print("Only TREND_FOLLOW and PULLBACK strategies")
        print("="*80)
        
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        print(f"\n📅 Period: {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"💰 Initial: ${self.initial_capital:,.0f}\n")
        
        signals_count = 0
        trades_count = 0
        
        for i, date in enumerate(all_dates):
            # Calculate equity
            total_equity = self.capital
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
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'positions': len(self.positions)
            })
            
            # Process exits
            for exit in self.check_exits(date):
                self.execute_exit(exit)
            
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
            if (i + 1) % 200 == 0:
                print(f"Day {i+1}/{len(all_dates)} | Equity: ${total_equity:,.0f} | Signals: {signals_count} | Trades: {trades_count}")
        
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
        print("BACKTEST RESULTS - V4.9 WINNERS ONLY")
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
        trades_df.to_csv('v49_winners_only_trades.csv', index=False)
        equity_df.to_csv('v49_winners_only_equity.csv', index=False)
        
        # Monthly/weekly
        self.create_summaries(equity_df, trades_df)
        
        print(f"\n📁 Files saved:")
        print(f"   • v49_winners_only_trades.csv - All {len(trades_df)} trades")
        print(f"   • v49_winners_only_equity.csv")
        print(f"   • v49_winners_only_monthly.csv")
        print(f"   • v49_winners_only_weekly.csv")
        
        # Comparison with V4.8
        print("\n" + "="*80)
        print("COMPARISON: V4.9 vs V4.8")
        print("="*80)
        print("V4.8 (6 strategies): $975K | 77.3% CAGR | -63.3% DD | 384 trades")
        print(f"V4.9 (2 strategies): ${final_equity:,.0f} | {cagr:.1f}% CAGR | {max_dd:.1f}% DD | {len(trades_df)} trades")
    
    def create_summaries(self, equity_df, trades_df):
        """Create summaries"""
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # Monthly
        monthly = equity_df.set_index('date').resample('M').last().reset_index()
        monthly['month'] = monthly['date'].dt.strftime('%Y-%m')
        monthly['profit'] = monthly['equity'].diff()
        monthly['profit_pct'] = monthly['equity'].pct_change() * 100
        monthly[['month', 'equity', 'profit', 'profit_pct']].to_csv('v49_winners_only_monthly.csv', index=False)
        
        # Weekly
        weekly = equity_df.set_index('date').resample('W').last().reset_index()
        weekly['week'] = weekly['date'].dt.strftime('%Y-W%U')
        weekly['profit'] = weekly['equity'].diff()
        weekly['profit_pct'] = weekly['equity'].pct_change() * 100
        weekly[['week', 'equity', 'profit', 'profit_pct']].to_csv('v49_winners_only_weekly.csv', index=False)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("V4.9 - WINNERS ONLY")
    print("Streamlined to use only the 2 winning strategies from V4.8")
    print("TREND_FOLLOW + PULLBACK = $1.18M profit in V4.8")
    print("="*80)
    
    strategy = V49WinnersOnly(initial_capital=10000)
    strategy.load_data(start_date='2025-06-01')
    strategy.run_backtest()

    
    print("\n✅ Complete!")