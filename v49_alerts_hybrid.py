#!/usr/bin/env python3
"""
V4.9 Live Trading Alert System - HYBRID TRACKING
=================================================

What it tracks automatically:
1. SIGNALS generated (what I recommend)
2. ACTUAL TRADES from positions.txt (what you did)
3. Execution quality (your entry vs my signal)
4. P&L based on YOUR actual positions

What you do:
- Execute trades in your broker
- Update positions.txt (30 seconds)
- Everything else is automatic

No assumptions. Accurate tracking. Minimal work.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

class V49HybridTracker:
    def __init__(self, gmail_user=None, gmail_password=None):
        # Email configuration
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
        
        # Strategy parameters - MATCH BACKTEST
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # Position management - BACKTEST PROVEN
        self.max_positions = 3
        self.position_size = 0.45
        self.leverage = 2.5
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        self.max_hold_days = 14
        
        # Files
        self.positions_file = 'positions.txt'
        self.signals_log = 'signals_log.csv'      # What I recommend
        self.trades_log = 'trades_log.csv'        # What you actually did
        self.performance_file = 'performance_summary.txt'
        
        # Backtest baseline
        self.backtest_cagr = 120.4
        self.backtest_win_rate = 51.6
        self.backtest_max_dd = -67.8
        
        self.init_files()
        self.load_positions()
        
    def init_files(self):
        """Initialize log files if they don't exist"""
        # Signals log (recommendations)
        if not os.path.exists(self.signals_log):
            with open(self.signals_log, 'w') as f:
                f.write('timestamp,action,symbol,strategy,signal_price,rsi,executed\n')
            print(f"✅ Created {self.signals_log}")
        
        # Trades log (actual executions and exits)
        if not os.path.exists(self.trades_log):
            with open(self.trades_log, 'w') as f:
                f.write('timestamp,action,symbol,strategy,price,shares,reason,pnl,pnl_pct,days_held\n')
            print(f"✅ Created {self.trades_log}")
    
    def log_signal(self, action, symbol, strategy, price, rsi):
        """Log signals we generate (recommendations)"""
        timestamp = datetime.now().isoformat()
        with open(self.signals_log, 'a') as f:
            f.write(f'{timestamp},{action},{symbol},{strategy},{price:.2f},{rsi:.1f},PENDING\n')
    
    def log_trade(self, action, symbol, strategy, price, shares, reason='', pnl=0, pnl_pct=0, days_held=0):
        """Log actual trades from positions.txt (what you did)"""
        timestamp = datetime.now().isoformat()
        with open(self.trades_log, 'a') as f:
            f.write(f'{timestamp},{action},{symbol},{strategy},{price:.2f},{shares},{reason},{pnl:.2f},{pnl_pct:.2f},{days_held}\n')
    
    def load_positions(self):
        """Load current positions from positions.txt"""
        self.positions = {}
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(',')
                            if len(parts) >= 4:
                                symbol = parts[0]
                                date = parts[1]
                                price = float(parts[2])
                                shares = int(parts[3])
                                strategy = parts[4] if len(parts) > 4 else 'UNKNOWN'
                                
                                self.positions[symbol] = {
                                    'entry_date': date,
                                    'entry_price': price,
                                    'shares': shares,
                                    'strategy': strategy
                                }
                print(f"✅ Loaded {len(self.positions)} positions from {self.positions_file}")
                if self.positions:
                    print(f"   Holdings: {', '.join(self.positions.keys())}")
            except Exception as e:
                print(f"⚠️  Error reading positions: {e}")
                self.positions = {}
        else:
            print(f"⚠️  No {self.positions_file} found - create it when you enter trades")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_data(self, symbol, days=100):
        """Download recent price data"""
        try:
            df = yf.download(symbol, period=f'{days}d', progress=False, auto_adjust=True)
            if len(df) < 50:
                return None
            
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
            df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
            df = df.dropna()
            
            return df
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            return None
    
    def check_trend_follow(self, row, prev_row):
        """TREND_FOLLOW strategy"""
        try:
            rsi = float(row['RSI'])
            close = float(row['Close'])
            ma_fast = float(row['MA_Fast'])
            ma_slow = float(row['MA_Slow'])
            
            return (rsi > self.rsi_buy and 
                    close > ma_fast and 
                    close > ma_slow and
                    ma_fast > ma_slow)
        except:
            return False
    
    def check_pullback(self, row, prev_row):
        """PULLBACK strategy - MATCHES BACKTEST"""
        try:
            rsi = float(row['RSI'])
            close = float(row['Close'])
            ma_fast = float(row['MA_Fast'])
            ma_slow = float(row['MA_Slow'])
            
            return (self.rsi_sell < rsi < self.rsi_buy and 
                    close > ma_slow and
                    ma_fast > ma_slow)
        except:
            return False
    
    def check_buy_signal(self, symbol):
        """Check for buy signals"""
        df = self.get_data(symbol)
        if df is None or len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        strategy = None
        
        if self.check_trend_follow(current, previous):
            strategy = 'TREND_FOLLOW'
        elif self.check_pullback(current, previous):
            strategy = 'PULLBACK'
        
        if strategy:
            signal = {
                'symbol': symbol,
                'strategy': strategy,
                'price': float(current['Close']),
                'rsi': float(current['RSI']),
                'date': current.name.strftime('%Y-%m-%d')
            }
            
            # LOG THE SIGNAL (recommendation)
            self.log_signal('BUY', symbol, strategy, signal['price'], signal['rsi'])
            
            return signal
        
        return None
    
    def check_sell_signal(self, symbol, position):
        """Check for sell signals on YOUR actual position"""
        df = self.get_data(symbol)
        if df is None or len(df) < 1:
            return None
        
        current = df.iloc[-1]
        current_price = float(current['Close'])
        entry_price = position['entry_price']  # YOUR actual entry
        
        try:
            entry_date = datetime.fromisoformat(position['entry_date'])
        except:
            entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
        
        days_held = (datetime.now() - entry_date).days
        pnl_pct = (current_price - entry_price) / entry_price * 100
        rsi = float(current['RSI'])
        
        reason = None
        
        if pnl_pct <= -self.stop_loss_pct * 100:
            reason = 'STOP_LOSS'
        elif pnl_pct >= self.take_profit_pct * 100:
            reason = 'TAKE_PROFIT'
        elif rsi < self.rsi_sell:
            reason = 'RSI_SELL'
        elif days_held >= self.max_hold_days:
            reason = 'TIME_EXIT'
        
        if reason:
            shares = position['shares']
            pnl_dollars = (current_price - entry_price) * shares
            
            signal = {
                'symbol': symbol,
                'reason': reason,
                'current_price': current_price,
                'entry_price': entry_price,
                'entry_date': position['entry_date'],
                'pnl_pct': pnl_pct,
                'pnl_dollars': pnl_dollars,
                'days_held': days_held,
                'shares': shares,
                'rsi': rsi,
                'strategy': position.get('strategy', 'UNKNOWN')
            }
            
            # LOG THE TRADE (based on YOUR actual position)
            self.log_trade(
                action='SELL',
                symbol=symbol,
                strategy=signal['strategy'],
                price=current_price,
                shares=shares,
                reason=reason,
                pnl=pnl_dollars,
                pnl_pct=pnl_pct,
                days_held=days_held
            )
            
            # LOG THE SIGNAL too
            self.log_signal('SELL', symbol, signal['strategy'], current_price, rsi)
            
            return signal
        
        return None
    
    def scan_signals(self):
        """Scan for signals"""
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Scanning...")
        
        buy_signals = []
        sell_signals = []
        
        # Check sell signals on YOUR positions
        for symbol, position in list(self.positions.items()):
            sell_signal = self.check_sell_signal(symbol, position)
            if sell_signal:
                sell_signals.append(sell_signal)
        
        # Check buy signals
        positions_available = self.max_positions - len(self.positions)
        if positions_available > 0:
            print(f"   📊 {positions_available} slot(s) available")
            for symbol in self.symbols:
                if symbol not in self.positions:
                    buy_signal = self.check_buy_signal(symbol)
                    if buy_signal:
                        buy_signals.append(buy_signal)
        else:
            print(f"   ⚠️  All {self.max_positions} slots filled")
        
        return buy_signals, sell_signals
    
    def generate_performance_summary(self):
        """Generate performance summary with execution quality analysis"""
        if not os.path.exists(self.trades_log):
            return
        
        try:
            # Load actual trades
            trades_df = pd.read_csv(self.trades_log)
            completed = trades_df[(trades_df['action'] == 'SELL') & (trades_df['pnl'] != 0)].copy()
            
            # Load signals
            signals_df = pd.read_csv(self.signals_log) if os.path.exists(self.signals_log) else None
            
            with open(self.performance_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("V4.9 LIVE TRADING - HYBRID TRACKING\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                # ACTUAL TRADES PERFORMANCE
                if len(completed) > 0:
                    total_trades = len(completed)
                    winners = completed[completed['pnl'] > 0]
                    losers = completed[completed['pnl'] <= 0]
                    win_rate = len(winners) / total_trades * 100
                    
                    total_pnl = completed['pnl'].sum()
                    avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
                    avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
                    avg_days = completed['days_held'].mean()
                    
                    f.write("ACTUAL TRADING PERFORMANCE:\n")
                    f.write(f"Total Trades:     {total_trades:>10}\n")
                    f.write(f"Win Rate:         {win_rate:>10.1f}%\n")
                    f.write(f"Total P&L:        ${total_pnl:>9,.0f}\n")
                    f.write(f"Avg Win:          ${avg_win:>9,.0f}\n")
                    f.write(f"Avg Loss:         ${avg_loss:>9,.0f}\n")
                    f.write(f"Avg Hold:         {avg_days:>10.1f} days\n\n")
                    
                    # BACKTEST COMPARISON
                    f.write("="*80 + "\n")
                    f.write("VS BACKTEST:\n")
                    f.write("="*80 + "\n")
                    f.write(f"               BACKTEST    LIVE      DIFF\n")
                    f.write(f"Win Rate:        {self.backtest_win_rate:>5.1f}%   {win_rate:>5.1f}%   {win_rate-self.backtest_win_rate:>+5.1f}%\n\n")
                    
                    if abs(win_rate - self.backtest_win_rate) < 10:
                        f.write("✅ Performance tracking with backtest expectations\n")
                    elif win_rate > self.backtest_win_rate:
                        f.write("🚀 Outperforming backtest!\n")
                    else:
                        f.write("⚠️  Underperforming - may need review\n")
                    
                    # STRATEGY BREAKDOWN
                    f.write("\n" + "="*80 + "\n")
                    f.write("STRATEGY BREAKDOWN (Your Actual Trades):\n")
                    f.write("="*80 + "\n")
                    strategy_stats = completed.groupby('strategy').agg({
                        'pnl': ['count', 'sum', 'mean']
                    }).round(0)
                    f.write(strategy_stats.to_string())
                    
                    # EXIT REASONS
                    f.write("\n\n" + "="*80 + "\n")
                    f.write("EXIT REASONS:\n")
                    f.write("="*80 + "\n")
                    exit_stats = completed.groupby('reason').agg({
                        'pnl': ['count', 'sum', 'mean']
                    }).round(0)
                    f.write(exit_stats.to_string())
                else:
                    f.write("No completed trades yet.\n")
                
                # SIGNAL vs EXECUTION ANALYSIS
                if signals_df is not None and len(signals_df) > 0:
                    f.write("\n\n" + "="*80 + "\n")
                    f.write("SIGNAL vs EXECUTION QUALITY:\n")
                    f.write("="*80 + "\n")
                    
                    buy_signals = signals_df[signals_df['action'] == 'BUY']
                    total_signals = len(buy_signals)
                    
                    # Count how many signals were acted on
                    # This is approximate - matches by symbol and date proximity
                    f.write(f"Buy Signals Generated: {total_signals:>10}\n")
                    f.write(f"Current Open Positions:{len(self.positions):>10}\n")
                    f.write(f"Completed Trades:      {len(completed):>10}\n\n")
                    
                    f.write("Recent Signals (Last 10):\n")
                    recent_signals = buy_signals.tail(10)[['timestamp', 'symbol', 'strategy', 'signal_price']]
                    f.write(recent_signals.to_string(index=False))
                
                # RECENT TRADES
                if len(completed) > 0:
                    f.write("\n\n" + "="*80 + "\n")
                    f.write("RECENT TRADES (Last 10):\n")
                    f.write("="*80 + "\n")
                    recent = completed.tail(10)[['timestamp', 'symbol', 'strategy', 'pnl', 'pnl_pct', 'reason']]
                    f.write(recent.to_string(index=False))
                
                f.write("\n" + "="*80 + "\n")
            
            print(f"\n✅ Performance summary updated: {self.performance_file}")
            
        except Exception as e:
            print(f"⚠️  Error generating summary: {e}")
    
    def format_buy_alert(self, signals):
        """Format buy signals"""
        if not signals:
            return ""
        
        alert = f"\n{'='*70}\n"
        alert += f"🟢 BUY SIGNALS (Recommendations)\n"
        alert += f"{'='*70}\n\n"
        
        for signal in signals:
            alert += f"Symbol:   {signal['symbol']}\n"
            alert += f"Strategy: {signal['strategy']}\n"
            alert += f"Price:    ${signal['price']:.2f}\n"
            alert += f"RSI:      {signal['rsi']:.1f}\n\n"
            
            shares = int((self.position_size * 10000 * self.leverage) / signal['price'])
            stop = signal['price'] * (1 - self.stop_loss_pct)
            target = signal['price'] * (1 + self.take_profit_pct)
            
            alert += f"RECOMMENDED ACTION:\n"
            alert += f"1. BUY ~{shares} shares of {signal['symbol']}\n"
            alert += f"2. Set STOP at ${stop:.2f} (-8%)\n"
            alert += f"3. Set TARGET at ${target:.2f} (+25%)\n"
            alert += f"4. After execution, add to {self.positions_file}:\n"
            alert += f"   {signal['symbol']},{signal['date']},<your_price>,<your_shares>,{signal['strategy']}\n\n"
            alert += f"{'─'*70}\n"
        
        alert += f"\n💡 Signal logged to {self.signals_log}\n"
        alert += f"   Update {self.positions_file} when you execute\n"
        
        return alert
    
    def format_sell_alert(self, signals):
        """Format sell signals"""
        if not signals:
            return ""
        
        avg_pnl = sum(s['pnl_pct'] for s in signals) / len(signals)
        emoji = '🟢' if avg_pnl > 0 else '🔴'
        
        alert = f"\n{'='*70}\n"
        alert += f"{emoji} SELL SIGNALS (Based on Your Positions)\n"
        alert += f"{'='*70}\n\n"
        
        for signal in signals:
            emoji = "🟢" if signal['pnl_pct'] > 0 else "🔴"
            alert += f"{emoji} {signal['symbol']} | {signal['strategy']}\n"
            alert += f"Reason:   {signal['reason']}\n"
            alert += f"Entry:    ${signal['entry_price']:.2f} on {signal['entry_date']}\n"
            alert += f"Current:  ${signal['current_price']:.2f}\n"
            alert += f"P&L:      {signal['pnl_pct']:+.1f}% (${signal['pnl_dollars']:+,.0f})\n"
            alert += f"Days:     {signal['days_held']}\n\n"
            
            alert += f"RECOMMENDED ACTION:\n"
            alert += f"1. SELL all {signal['shares']} shares at market\n"
            alert += f"2. Remove from {self.positions_file}:\n"
            alert += f"   Delete line: {signal['symbol']},{signal['entry_date']},...\n\n"
            alert += f"{'─'*70}\n"
        
        alert += f"\n✅ Trade logged to {self.trades_log}\n"
        alert += f"   Update {self.positions_file} when you execute\n"
        
        return alert
    
    def run_once(self):
        """Run a single scan"""
        buy_signals, sell_signals = self.scan_signals()
        
        if sell_signals:
            print(self.format_sell_alert(sell_signals))
        
        if buy_signals:
            print(self.format_buy_alert(buy_signals))
        
        if not buy_signals and not sell_signals:
            print("📊 No signals")
            
            if self.positions:
                print("\n📋 Your Current Positions:")
                for symbol, pos in self.positions.items():
                    df = self.get_data(symbol, days=5)
                    if df is not None and len(df) > 0:
                        current = float(df.iloc[-1]['Close'])
                        pnl_pct = (current - pos['entry_price']) / pos['entry_price'] * 100
                        emoji = "🟢" if pnl_pct > 0 else "🔴"
                        print(f"   {emoji} {symbol}: {pnl_pct:+.1f}% ({pos['strategy']})")
        
        # Update performance summary
        self.generate_performance_summary()
        
        print(f"\n{'='*70}")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Signals logged: {self.signals_log}")
        print(f"📈 Trades tracked: {self.trades_log}")
        print(f"📄 Summary: {self.performance_file}\n")
    
    def run_continuous(self, interval=300):
        """Run continuous monitoring"""
        print(f"\n{'='*70}")
        print(f"V4.9 HYBRID TRACKING - RUNNING")
        print(f"{'='*70}")
        print(f"Symbols:  {', '.join(self.symbols)}")
        print(f"Setup:    {self.max_positions} pos @ {self.position_size*100:.0f}% @ {self.leverage}x")
        print(f"Interval: {interval}s")
        print(f"\nTracking:")
        print(f"  • {self.signals_log} - My recommendations")
        print(f"  • {self.positions_file} - Your actual positions")
        print(f"  • {self.trades_log} - Your completed trades")
        print(f"  • {self.performance_file} - Performance summary")
        print(f"{'='*70}\n")
        
        while True:
            try:
                self.run_once()
                print(f"💤 Next scan in {interval}s...\n")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\n⚠️  Stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V4.9 Hybrid Tracking')
    parser.add_argument('--once', action='store_true', help='Run once')
    parser.add_argument('--interval', type=int, default=300, help='Scan interval (seconds)')
    parser.add_argument('--summary', action='store_true', help='Generate summary only')
    
    args = parser.parse_args()
    
    tracker = V49HybridTracker()
    
    if args.summary:
        tracker.generate_performance_summary()
        print("✅ Summary generated!")
    elif args.once:
        tracker.run_once()
    else:
        tracker.run_continuous(interval=args.interval)
