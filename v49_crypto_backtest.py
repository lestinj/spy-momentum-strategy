#!/usr/bin/env python3
"""
V4.9 CRYPTO EDITION
==================

Same winning strategy (TREND_FOLLOW + PULLBACK) applied to cryptocurrency.

Adaptations for crypto:
- Lower leverage (1.5x vs 2.5x) due to higher volatility
- Slightly wider stops (10% vs 8%)
- Higher take profit (30% vs 25%)
- Same 14-day max hold
- 24/7 market (more signals)

Testing on 7 high-quality crypto assets.
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
        
        # Crypto universe - Yahoo Finance format
        # Using -USD pairs that yfinance supports
        self.symbols = [
            'BTC-USD',   # Bitcoin - Blue chip
            'ETH-USD',   # Ethereum - Blue chip
            'SOL-USD',   # Solana - High quality L1
            'AVAX-USD',  # Avalanche - High quality L1
            'LINK-USD',  # Chainlink - Oracle leader
            'MATIC-USD', # Polygon - Scaling solution
            'DOT-USD'    # Polkadot - Interoperability
        ]
        
        # V4.9 strategy parameters - ADAPTED FOR CRYPTO
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # Position management - CRYPTO ADJUSTMENTS
        self.max_positions = 3
        self.position_size = 0.35  # 35% per position
        self.leverage = 1.5        # Lower than stocks (crypto is 2x more volatile)
        self.stop_loss_pct = 0.10  # 10% (wider due to crypto noise)
        self.take_profit_pct = 0.30  # 30% (crypto moves more)
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
        """Load crypto data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print("V4.9 CRYPTO EDITION - Data Loading")
        print(f"{'='*80}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Strategy: TREND_FOLLOW + PULLBACK (adapted for crypto)")
        print(f"\n📊 Downloading crypto data from Yahoo Finance...\n")
        
        self.data = {}
        data_quality = []
        
        for symbol in self.symbols:
            try:
                # Download with auto_adjust to avoid warnings
                df = yf.download(symbol, start=start_date, end=end_date, 
                                progress=False, auto_adjust=True)
                
                if len(df) > 100:
                    # Calculate indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    # Calculate daily volatility for comparison
                    daily_returns = df['Close'].pct_change()
                    volatility = daily_returns.std() * np.sqrt(365) * 100
                    
                    self.data[symbol] = df
                    
                    # Clean symbol name for display
                    clean_name = symbol.replace('-USD', '')
                    
                    print(f"✓ {clean_name:6} {len(df):>5} days | Vol: {volatility:>5.1f}% | "
                          f"Range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
                    
                    data_quality.append({
                        'symbol': symbol,
                        'days': len(df),
                        'volatility': volatility,
                        'quality': 'Good' if len(df) > 500 else 'Limited'
                    })
                else:
                    print(f"✗ {symbol}: Insufficient data ({len(df)} days)")
                    
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
        
        print(f"\n✓ Successfully loaded {len(self.data)} crypto assets")
        
        # Data quality summary
        if data_quality:
            avg_vol = np.mean([d['volatility'] for d in data_quality])
            print(f"\n📊 Average crypto volatility: {avg_vol:.1f}% annually")
            print(f"    (Compare to TSLA ~80%, NVDA ~70%)")
            print(f"\n{'='*80}\n")
        
        return len(self.data) > 0
    
    def generate_signals(self, date):
        """Generate trading signals - SAME LOGIC AS V4.9 STOCKS"""
        signals = []
        
        for symbol, df in self.data.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 50:
                continue
            
            try:
                current = df.loc[date]
                
                rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                close = float(current['Close'])
                ma_fast = float(current['MA_Fast']) if not pd.isna(current['MA_Fast']) else 0
                ma_slow = float(current['MA_Slow']) if not pd.isna(current['MA_Slow']) else 0
                
                if rsi == 0 or ma_fast == 0 or ma_slow == 0:
                    continue
                
                strategy_triggered = None
                quality = 1
                
                # STRATEGY 1: TREND_FOLLOW (same as stocks)
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    strategy_triggered = 'TREND_FOLLOW'
                    quality = 3
                
                # STRATEGY 2: PULLBACK (same as stocks)
                elif (rsi < self.rsi_buy and rsi > 45 and
                      close > ma_slow and
                      ma_fast > ma_slow):
                    strategy_triggered = 'PULLBACK'
                    quality = 3
                
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
        """Check exit conditions - ADAPTED STOPS FOR CRYPTO"""
        exits = []
        
        for symbol, position in list(self.positions.items()):
            if date not in self.data[symbol].index:
                continue
            
            try:
                current = self.data[symbol].loc[date]
                current_price = float(current['Close'])
                current_rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                
                entry_price = position['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price)
                days_held = (date - position['entry_date']).days
                
                exit_reason = None
                
                # Stop loss (10% for crypto vs 8% for stocks)
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = 'STOP_LOSS'
                
                # Take profit (30% for crypto vs 25% for stocks)
                elif pnl_pct >= self.take_profit_pct:
                    exit_reason = 'TAKE_PROFIT'
                
                # RSI exit
                elif current_rsi > 0 and current_rsi < self.rsi_sell:
                    exit_reason = 'RSI_SELL'
                
                # Time exit
                elif days_held >= self.max_hold_days:
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
        
        # Calculate position size with leverage
        position_value = self.capital * self.position_size
        leveraged_value = position_value * self.leverage
        shares = leveraged_value / signal['price']
        
        if shares > 0:
            self.positions[signal['symbol']] = {
                'entry_price': signal['price'],
                'entry_date': signal['date'],
                'shares': shares,
                'entry_value': position_value,
                'leveraged_value': leveraged_value,
                'strategy': signal['strategy']
            }
            
            self.capital -= position_value
            
            clean_name = signal['symbol'].replace('-USD', '')
            print(f"🟢 BUY  {clean_name:6} ${signal['price']:>10,.2f} | "
                  f"{signal['strategy']:13} | RSI: {signal['rsi']:.1f}")
    
    def execute_exit(self, exit):
        """Execute sell trade"""
        symbol = exit['symbol']
        exit_price = exit['exit_price']
        exit_date = exit['date']
        exit_reason = exit['exit_reason']
        
        position = self.positions[symbol]
        
        current_value = position['shares'] * exit_price
        pnl = current_value - position['leveraged_value']
        pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        
        self.capital += position['entry_value'] + pnl
        
        self.trades.append({
            'symbol': symbol,
            'strategy': position['strategy'],
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'exit_date': exit_date,
            'exit_price': exit_price,
            'shares': position['shares'],
            'entry_value': position['entry_value'],
            'exit_value': current_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': (exit_date - position['entry_date']).days,
            'exit_reason': exit_reason
        })
        
        del self.positions[symbol]
        
        emoji = '🟢' if pnl > 0 else '🔴'
        clean_name = symbol.replace('-USD', '')
        print(f"{emoji} SELL {clean_name:6} ${exit_price:>10,.2f} | "
              f"{exit_reason:13} | {pnl_pct:+6.1f}% | "
              f"${pnl:+,.0f}")
    
    def run_backtest(self):
        """Run backtest"""
        print("\n" + "="*80)
        print("V4.9 CRYPTO BACKTEST - RUNNING")
        print("="*80)
        
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        print(f"\n📅 Period: {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.0f}")
        print(f"📊 Strategy: TREND_FOLLOW + PULLBACK")
        print(f"🎯 Position Setup: {self.max_positions} positions @ {self.position_size*100:.0f}% @ {self.leverage}x leverage")
        print(f"🛡️  Risk: {self.stop_loss_pct*100:.0f}% stop | {self.take_profit_pct*100:.0f}% target | {self.max_hold_days}d max\n")
        
        signals_count = 0
        trades_count = 0
        
        for i, date in enumerate(all_dates):
            # Calculate equity
            total_equity = self.capital
            for symbol, pos in self.positions.items():
                if date in self.data[symbol].index:
                    try:
                        current_price = float(self.data[symbol].loc[date]['Close'])
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
                
                signals = sorted(signals, key=lambda x: x['quality'], reverse=True)
                
                for signal in signals[:self.max_positions - len(self.positions)]:
                    if signal['symbol'] not in self.positions:
                        self.execute_trade(signal)
                        trades_count += 1
            
            # Progress
            if (i + 1) % 365 == 0:
                years = (i + 1) / 365
                print(f"\n📆 Year {years:.1f} | Equity: ${total_equity:,.0f} | "
                      f"Trades: {trades_count} | Signals: {signals_count}")
        
        # Close remaining positions
        if self.positions:
            final_date = all_dates[-1]
            print(f"\n🔴 Closing {len(self.positions)} remaining positions:")
            for symbol in list(self.positions.keys()):
                if final_date in self.data[symbol].index:
                    try:
                        exit_price = float(self.data[symbol].loc[final_date]['Close'])
                        self.execute_exit({
                            'symbol': symbol,
                            'exit_price': exit_price,
                            'date': final_date,
                            'exit_reason': 'BACKTEST_END'
                        })
                    except:
                        continue
        
        print(f"\n📊 Total Signals: {signals_count} | Total Trades: {trades_count}")
        self.generate_reports()
    
    def generate_reports(self):
        """Generate performance reports"""
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
        print("V4.9 CRYPTO RESULTS")
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
        
        if len(losing) > 0 and losing['pnl'].sum() != 0:
            win_loss_ratio = abs(winning['pnl'].mean() / losing['pnl'].mean())
            profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum())
            print(f"Win/Loss Ratio:     {win_loss_ratio:>15.2f}x")
            print(f"Profit Factor:      {profit_factor:>15.2f}")
        
        print("="*80)
        
        # Compare to stocks
        print("\n" + "="*80)
        print("COMPARISON: CRYPTO vs STOCKS")
        print("="*80)
        print(f"V4.9 Stocks:        $971K | 120.4% CAGR | -67.8% DD | 353 trades")
        print(f"V4.9 Crypto:        ${final_equity:,.0f} | {cagr:.1f}% CAGR | {max_dd:.1f}% DD | {len(trades_df)} trades")
        
        if cagr > 120.4:
            diff = cagr - 120.4
            print(f"\n🚀 CRYPTO WINS by {diff:.1f}% CAGR!")
        elif cagr > 100:
            print(f"\n✅ CRYPTO STRONG - Similar performance to stocks")
        else:
            print(f"\n⚠️  STOCKS OUTPERFORMED - Consider sticking with equities")
        
        # Strategy breakdown
        print("\n" + "="*80)
        print("STRATEGY BREAKDOWN:")
        print("="*80)
        strategy_stats = trades_df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(0)
        print(strategy_stats)
        
        # Coin breakdown
        print("\n" + "="*80)
        print("COIN BREAKDOWN:")
        print("="*80)
        trades_df['clean_symbol'] = trades_df['symbol'].str.replace('-USD', '')
        coin_stats = trades_df.groupby('clean_symbol').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(0)
        print(coin_stats)
        
        # Exit reasons
        print("\n" + "="*80)
        print("EXIT REASON BREAKDOWN:")
        print("="*80)
        exit_stats = trades_df.groupby('exit_reason').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(0)
        print(exit_stats)
        
        # Save files
        trades_df.to_csv('v49_crypto_trades.csv', index=False)
        equity_df.to_csv('v49_crypto_equity.csv', index=False)
        
        print(f"\n📁 Files saved:")
        print(f"   • v49_crypto_trades.csv - All {len(trades_df)} trades")
        print(f"   • v49_crypto_equity.csv - Daily equity curve")
        
        print("\n" + "="*80)
        print("✅ CRYPTO BACKTEST COMPLETE")
        print("="*80)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                      V4.9 CRYPTO EDITION                              ║
    ║                                                                        ║
    ║  Same winning momentum strategies applied to cryptocurrency           ║
    ║                                                                        ║
    ║  Strategies:                                                          ║
    ║    • TREND_FOLLOW - Ride the momentum                                 ║
    ║    • PULLBACK - Buy dips in uptrends                                  ║
    ║                                                                        ║
    ║  Crypto Adaptations:                                                  ║
    ║    • Lower leverage (1.5x vs 2.5x)                                    ║
    ║    • Wider stops (10% vs 8%)                                          ║
    ║    • Higher targets (30% vs 25%)                                      ║
    ║    • Same 14-day max hold                                             ║
    ║                                                                        ║
    ║  Testing on: BTC, ETH, SOL, AVAX, LINK, MATIC, DOT                    ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    strategy = V49Crypto(initial_capital=10000)
    
    if strategy.load_data(start_date='2020-01-01'):
        strategy.run_backtest()
    else:
        print("\n❌ Failed to load sufficient crypto data")
        print("\n💡 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Try different date range (some coins newer)")
        print("   3. Update yfinance: pip install --upgrade yfinance")
        print("   4. For better crypto data, consider: CCXT, Binance API, or CryptoCompare")
    
    print("\n✅ Complete!")
