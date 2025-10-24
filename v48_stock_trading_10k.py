"""
V4.8 AGGRESSIVE STOCK STRATEGY
Starting Capital: $10,000
Target: 150-250% CAGR (matching original V4.8's 203% CAGR)

KEY CHANGES FROM CONSERVATIVE VERSION:
1. Much larger position sizes (20-30% per trade, not 6-10%)
2. Fewer positions (3-4 max, not 8) - concentration beats diversification
3. Only take highest quality signals (★★★ only)
4. More aggressive profit targets (20%, not 15%)
5. Shorter hold times (exit faster)
6. INCLUDES LIVE SIGNAL GENERATOR - shows what to trade TODAY
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class V48AggressiveStockStrategy:
    def __init__(self, leverage=2.0):
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.leverage = leverage
        
        # Core momentum stocks (fewer, higher quality)
        self.symbols = [
            'NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN',
            'SMCI', 'MSTR', 'RIOT', 'MARA'
        ]
        
        # AGGRESSIVE SETTINGS
        self.max_positions = 4  # Only 3-4 positions (concentration)
        self.min_quality = 3    # Only take ★★★ signals
        self.position_size_pct = 0.25  # 25% per position!
        self.stop_loss_pct = 0.06      # 6% stop
        self.take_profit_pct = 0.20    # 20% target
        self.max_hold_days = 15        # Exit by 15 days
        
        # Momentum parameters
        self.rsi_period = 14
        self.rsi_buy_threshold = 60  # Higher threshold
        self.ma_fast = 10
        self.ma_slow = 30
        
        self.trades = []
        self.open_positions = {}
        self.data_cache = {}
        
    def load_data(self, start_date='2020-01-01', end_date=None):
        """Load price data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\nLoading data for {len(self.symbols)} symbols...")
        print(f"Period: {start_date} to {end_date}\n")
        
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(df) > 100:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    df.columns = [str(c).lower() for c in df.columns]
                    self.data_cache[symbol] = df
                    print(f"Loading {symbol}... ✓ {len(df)} days")
                else:
                    print(f"Loading {symbol}... ✗ Insufficient data")
            except Exception as e:
                print(f"Loading {symbol}... ✗ Error: {e}")
        
        print(f"\n✓ Successfully loaded {len(self.data_cache)}/{len(self.symbols)} symbols\n")
        
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['ma_fast'] = df['close'].rolling(window=self.ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(window=self.ma_slow).mean()
        df['ma_200'] = df['close'].rolling(window=200).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price strength
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        return df
    
    def generate_signals(self, symbol, df, current_date):
        """Generate HIGH QUALITY signals only"""
        signals = []
        
        if current_date not in df.index:
            return signals
        
        idx = df.index.get_loc(current_date)
        if idx < 200:  # Need 200 days for MA200
            return signals
        
        current = df.iloc[idx]
        
        # ONLY ★★★ QUALITY SIGNALS
        
        # 1. Strong Momentum Breakout (★★★)
        high_20 = df['high'].iloc[idx-20:idx].max()
        if (current['close'] > high_20 and
            current['close'] > current['ma_200'] and
            current['rsi'] > 65 and
            current['volume_ratio'] > 1.5 and
            current['price_change_20d'] > 0.15):  # Up 15%+ in 20 days
            signals.append({
                'symbol': symbol,
                'strategy': 'MOMENTUM_BREAKOUT',
                'quality': 3,
                'strength': 10
            })
        
        # 2. Explosive Trend Following (★★★)
        if (current['ma_fast'] > current['ma_slow'] and
            current['ma_slow'] > current['ma_200'] and
            current['close'] > current['ma_fast'] and
            current['rsi'] > 70 and
            df['rsi'].iloc[idx-1] > df['rsi'].iloc[idx-2] and
            current['volume_ratio'] > 1.3 and
            current['price_change_5d'] > 0.08):  # Up 8%+ in 5 days
            signals.append({
                'symbol': symbol,
                'strategy': 'EXPLOSIVE_TREND',
                'quality': 3,
                'strength': 9
            })
        
        # 3. Volume Surge Momentum (★★★)
        if (current['volume_ratio'] > 2.5 and
            current['close'] > df['close'].iloc[idx-1] * 1.03 and  # Up 3%+ today
            current['rsi'] > 60 and
            current['close'] > current['ma_200']):
            signals.append({
                'symbol': symbol,
                'strategy': 'VOLUME_SURGE',
                'quality': 3,
                'strength': 8
            })
        
        return signals
    
    def calculate_position_size(self, price):
        """Calculate shares for aggressive 25% position"""
        leveraged_capital = self.current_capital * self.leverage
        position_value = leveraged_capital * self.position_size_pct
        shares = int(position_value / price)
        shares = max(1, shares)
        actual_cost = shares * price
        
        if actual_cost > self.current_capital:
            shares = int(self.current_capital / price)
            actual_cost = shares * price
        
        return shares, actual_cost
    
    def enter_position(self, symbol, date, price, quality, strategy, strength):
        """Enter aggressive position"""
        if len(self.open_positions) >= self.max_positions:
            return None
        
        if symbol in self.open_positions:
            return None
        
        shares, cost = self.calculate_position_size(price)
        
        if shares < 1 or cost > self.current_capital:
            return None
        
        stop_loss = price * (1 - self.stop_loss_pct)
        take_profit = price * (1 + self.take_profit_pct)
        
        position = {
            'symbol': symbol,
            'entry_date': date,
            'entry_price': price,
            'shares': shares,
            'cost': cost,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quality': quality,
            'strategy': strategy,
            'strength': strength
        }
        
        self.open_positions[symbol] = position
        self.current_capital -= cost
        
        return position
    
    def check_exits(self, current_date):
        """Check exits with aggressive management"""
        to_exit = []
        
        for symbol, position in self.open_positions.items():
            if symbol not in self.data_cache:
                continue
            
            df = self.data_cache[symbol]
            if current_date not in df.index:
                continue
            
            current_price = df.loc[current_date, 'close']
            days_held = (current_date - position['entry_date']).days
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                to_exit.append((symbol, current_price, 'STOP_LOSS'))
            
            # Check take profit
            elif current_price >= position['take_profit']:
                to_exit.append((symbol, current_price, 'TAKE_PROFIT'))
            
            # Trailing stop (protect profits)
            elif current_price > position['entry_price'] * 1.10:  # Up 10%
                trailing_stop = current_price * 0.95  # 5% trailing
                if current_price < trailing_stop:
                    to_exit.append((symbol, current_price, 'TRAILING_STOP'))
            
            # Time exit
            elif days_held >= self.max_hold_days:
                to_exit.append((symbol, current_price, 'TIME_EXIT'))
        
        for symbol, exit_price, reason in to_exit:
            self.exit_position(symbol, current_date, exit_price, reason)
    
    def exit_position(self, symbol, date, price, reason):
        """Exit position"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        proceeds = position['shares'] * price
        pnl = proceeds - position['cost']
        return_pct = (pnl / position['cost']) * 100
        
        self.current_capital += proceeds
        
        trade = {
            'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': date.strftime('%Y-%m-%d'),
            'symbol': symbol,
            'strategy': position['strategy'],
            'quality': position['quality'],
            'strength': position['strength'],
            'shares': position['shares'],
            'entry_price': round(position['entry_price'], 2),
            'exit_price': round(price, 2),
            'cost': round(position['cost'], 2),
            'proceeds': round(proceeds, 2),
            'pnl': round(pnl, 2),
            'return_pct': round(return_pct, 2),
            'days_held': (date - position['entry_date']).days,
            'exit_reason': reason
        }
        
        self.trades.append(trade)
        del self.open_positions[symbol]
    
    def run_backtest(self):
        """Run backtest"""
        if not self.data_cache:
            print("No data loaded!")
            return
        
        all_dates = set()
        for df in self.data_cache.values():
            all_dates.update(df.index)
        all_dates = sorted(list(all_dates))
        
        print("=" * 80)
        print("RUNNING V4.8 AGGRESSIVE STOCK BACKTEST")
        print("=" * 80)
        print(f"Starting Capital: ${self.initial_capital:,.0f}")
        print(f"Leverage: {self.leverage}x")
        print(f"Effective Buying Power: ${self.initial_capital * self.leverage:,.0f}")
        print(f"Symbols: {len(self.data_cache)}")
        print(f"Max Positions: {self.max_positions} (CONCENTRATED)")
        print(f"Position Size: {self.position_size_pct*100:.0f}% per trade (AGGRESSIVE)")
        print(f"Quality Filter: ★★★ ONLY")
        print(f"Stop Loss: {self.stop_loss_pct*100:.0f}%")
        print(f"Take Profit: {self.take_profit_pct*100:.0f}%")
        print(f"Target CAGR: 150-250%")
        print("=" * 80)
        
        print("\nCalculating indicators...")
        for symbol in self.data_cache:
            self.data_cache[symbol] = self.calculate_indicators(self.data_cache[symbol])
        
        print("\nRunning simulation...\n")
        equity_curve = []
        
        for i, current_date in enumerate(all_dates[200:], 200):
            self.check_exits(current_date)
            
            if len(self.open_positions) < self.max_positions:
                all_signals = []
                
                for symbol in self.data_cache:
                    signals = self.generate_signals(symbol, self.data_cache[symbol], current_date)
                    all_signals.extend(signals)
                
                # Sort by strength
                all_signals.sort(key=lambda x: x['strength'], reverse=True)
                
                for signal in all_signals:
                    if len(self.open_positions) >= self.max_positions:
                        break
                    
                    symbol = signal['symbol']
                    if symbol in self.open_positions:
                        continue
                    
                    df = self.data_cache[symbol]
                    if current_date not in df.index:
                        continue
                    
                    price = df.loc[current_date, 'close']
                    self.enter_position(symbol, current_date, price,
                                      signal['quality'], signal['strategy'], signal['strength'])
            
            position_value = sum(
                pos['shares'] * self.data_cache[pos['symbol']].loc[current_date, 'close']
                for sym, pos in self.open_positions.items()
                if current_date in self.data_cache[sym].index
            )
            total_equity = self.current_capital + position_value
            
            equity_curve.append({
                'date': current_date,
                'equity': total_equity,
                'cash': self.current_capital,
                'positions': len(self.open_positions)
            })
            
            if i % 100 == 0:
                print(f"Progress: {current_date.strftime('%Y-%m-%d')} | "
                      f"Equity: ${total_equity:,.0f} | Positions: {len(self.open_positions)}")
        
        # Close remaining positions
        final_date = all_dates[-1]
        for symbol in list(self.open_positions.keys()):
            if symbol in self.data_cache and final_date in self.data_cache[symbol].index:
                final_price = self.data_cache[symbol].loc[final_date, 'close']
                self.exit_position(symbol, final_date, final_price, 'BACKTEST_END')
        
        self.equity_curve = pd.DataFrame(equity_curve)
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze results"""
        if not self.trades:
            print("\nNo trades executed!")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS - V4.8 AGGRESSIVE STOCKS")
        print("=" * 80)
        
        final_equity = self.equity_curve['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        days = (self.equity_curve['date'].iloc[-1] - self.equity_curve['date'].iloc[0]).days
        years = days / 365.25
        cagr = (((final_equity / self.initial_capital) ** (1 / years)) - 1) * 100
        
        self.equity_curve['peak'] = self.equity_curve['equity'].cummax()
        self.equity_curve['drawdown'] = (
            (self.equity_curve['equity'] - self.equity_curve['peak']) /
            self.equity_curve['peak'] * 100
        )
        max_drawdown = self.equity_curve['drawdown'].min()
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Starting Capital:    ${self.initial_capital:>12,.0f}")
        print(f"Ending Capital:      ${final_equity:>12,.0f}")
        print(f"Total Return:        {total_return:>12.1f}%")
        print(f"CAGR:               {cagr:>12.1f}%")
        print(f"Max Drawdown:        {max_drawdown:>12.1f}%")
        print(f"Leverage Used:       {self.leverage}x")
        
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(df_trades) * 100
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"Total Trades:        {len(df_trades):>12,}")
        print(f"Winning Trades:      {len(winning_trades):>12,}")
        print(f"Losing Trades:       {len(losing_trades):>12,}")
        print(f"Win Rate:            {win_rate:>12.1f}%")
        print(f"Avg Win:             ${winning_trades['pnl'].mean():>12,.2f}")
        print(f"Avg Loss:            ${losing_trades['pnl'].mean():>12,.2f}")
        print(f"Avg Hold:            {df_trades['days_held'].mean():>12.1f} days")
        
        print(f"\n🎯 TOP 5 TRADES:")
        top_trades = df_trades.nlargest(5, 'pnl')
        for _, trade in top_trades.iterrows():
            print(f"  {trade['symbol']}: ${trade['pnl']:>8,.2f} ({trade['return_pct']:>6.1f}%) - {trade['strategy']}")
        
        print("\n" + "=" * 80)
        
        self.save_results(df_trades)
    
    def save_results(self, df_trades):
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        trades_file = f'v48_aggressive_stock_trades_{timestamp}.csv'
        df_trades.to_csv(trades_file, index=False)
        print(f"\n✅ Trades saved to: {trades_file}")
        
        equity_file = f'v48_aggressive_stock_equity_{timestamp}.csv'
        self.equity_curve.to_csv(equity_file, index=False)
        print(f"✅ Equity curve saved to: {equity_file}")
        
        print(f"\n📋 SAMPLE TRADES (first 20):")
        print(df_trades.head(20).to_string(index=False))
    
    def generate_live_signals(self):
        """Generate trading signals for TODAY - THIS IS WHAT YOU TRADE!"""
        print("\n" + "=" * 80)
        print("🔴 LIVE TRADING SIGNALS FOR TODAY")
        print("=" * 80)
        
        if not self.data_cache:
            print("No data loaded!")
            return []
        
        # Calculate indicators first if not already done
        print("\nCalculating indicators...")
        for symbol in self.data_cache:
            # Check if indicators already calculated
            if 'ma_fast' not in self.data_cache[symbol].columns:
                self.data_cache[symbol] = self.calculate_indicators(self.data_cache[symbol])
        
        # Get most recent date
        latest_dates = [df.index[-1] for df in self.data_cache.values()]
        current_date = max(latest_dates)
        
        print(f"Signal Date: {current_date.strftime('%Y-%m-%d')}")
        print(f"Quality Filter: ★★★ ONLY")
        print(f"Position Size: {self.position_size_pct*100:.0f}% of capital (${self.initial_capital * self.leverage * self.position_size_pct:,.0f})")
        print("=" * 80)
        
        # Generate signals for each symbol
        live_signals = []
        
        for symbol in self.data_cache:
            df = self.data_cache[symbol]
            signals = self.generate_signals(symbol, df, current_date)
            
            if signals:
                current_price = df.loc[current_date, 'close']
                
                for signal in signals:
                    shares, cost = self.calculate_position_size(current_price)
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                    
                    live_signals.append({
                        'symbol': symbol,
                        'strategy': signal['strategy'],
                        'quality': signal['quality'],
                        'strength': signal['strength'],
                        'current_price': round(current_price, 2),
                        'shares': shares,
                        'cost': round(cost, 2),
                        'stop_loss': round(stop_loss, 2),
                        'take_profit': round(take_profit, 2)
                    })
        
        # Sort by strength
        live_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        if not live_signals:
            print("\n❌ NO SIGNALS TODAY - No trades to execute")
            print("Check again tomorrow!")
            return []
        
        print(f"\n✅ FOUND {len(live_signals)} SIGNAL(S) - EXECUTE THESE TRADES:\n")
        
        for i, signal in enumerate(live_signals[:self.max_positions], 1):
            print(f"{'='*80}")
            print(f"TRADE #{i} - {signal['strategy']} (Strength: {signal['strength']}/10)")
            print(f"{'='*80}")
            print(f"Symbol:           {signal['symbol']}")
            print(f"Action:           BUY")
            print(f"Current Price:    ${signal['current_price']}")
            print(f"Shares to Buy:    {signal['shares']}")
            print(f"Total Cost:       ${signal['cost']:,.2f}")
            print(f"Stop Loss:        ${signal['stop_loss']} (-{self.stop_loss_pct*100:.0f}%)")
            print(f"Take Profit:      ${signal['take_profit']} (+{self.take_profit_pct*100:.0f}%)")
            print(f"\n📱 EXECUTION STEPS:")
            print(f"1. Login to your broker")
            print(f"2. Search for: {signal['symbol']}")
            print(f"3. Action: BUY {signal['shares']} shares")
            print(f"4. Order Type: LIMIT at ${signal['current_price']}")
            print(f"5. Set STOP LOSS at ${signal['stop_loss']}")
            print(f"6. Set TAKE PROFIT at ${signal['take_profit']}")
            print()
        
        # Save signals to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        signals_file = f'live_signals_{timestamp}.csv'
        pd.DataFrame(live_signals).to_csv(signals_file, index=False)
        print(f"✅ Signals saved to: {signals_file}\n")
        
        return live_signals

def main():
    print("\n" + "=" * 80)
    print("V4.8 AGGRESSIVE STOCK STRATEGY - $10,000")
    print("TARGET: 150-250% CAGR (matching original V4.8's 203%)")
    print("=" * 80)
    
    print("\nSELECT MODE:")
    print("1. Run Backtest + Generate Today's Signals (Recommended)")
    print("2. Generate Today's Signals Only (Quick)")
    print("3. Compare Multiple Leverage Levels")
    
    choice = input("\nEnter choice (1-3) or press Enter for 1: ").strip()
    if choice == '':
        choice = '1'
    
    if choice == '3':
        # Compare leverage levels
        for lev in [1.0, 2.0, 4.0]:
            print(f"\n{'='*80}")
            print(f"TESTING {lev}X LEVERAGE")
            print(f"{'='*80}")
            strategy = V48AggressiveStockStrategy(leverage=lev)
            strategy.load_data(start_date='2020-01-01')
            strategy.run_backtest()
    else:
        # Default 2x leverage
        strategy = V48AggressiveStockStrategy(leverage=2.0)
        strategy.load_data(start_date='2020-01-01')
        
        if choice in ['1', '']:
            strategy.run_backtest()
        
        # Always show live signals
        print("\n" + "=" * 80)
        print("NOW GENERATING TODAY'S TRADING SIGNALS...")
        print("=" * 80)
        strategy.generate_live_signals()
    
    print("\n✅ Complete! You now know EXACTLY what to trade.")

if __name__ == "__main__":
    main()