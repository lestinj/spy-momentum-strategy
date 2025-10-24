"""
V49 MOMENTUM STRATEGY - WITH CIRCUIT BREAKER
═══════════════════════════════════════════════════════════════════════════════

NEW FEATURES:
1. ✅ Circuit Breaker - Stops trading during severe drawdowns
2. ✅ Configurable trigger levels (-15%, -20%, -25%, etc.)
3. ✅ Multiple protection modes (PAUSE, REDUCE, STOP)
4. ✅ Auto-resume when conditions improve
5. ✅ Tracks circuit breaker events

CIRCUIT BREAKER MODES:
- PAUSE:  Stop new trades, keep existing positions (conservative)
- REDUCE: Cut position sizes in half during drawdowns
- STOP:   Close all positions and stop trading (most protective)

GOAL: Reduce max drawdown while maintaining strong returns
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class V49WithCircuitBreaker:
    def __init__(self, initial_capital=20000, 
                 use_dynamic_sizing=True,
                 leverage=2.0,
                 position_size_pct=0.35,
                 max_positions=3,
                 use_circuit_breaker=False,
                 circuit_breaker_threshold=-0.20,  # -20% drawdown triggers
                 circuit_breaker_mode='PAUSE',     # PAUSE, REDUCE, or STOP
                 circuit_breaker_reset=-0.10):     # Resume at -10% drawdown
        """
        Initialize V49 with optional circuit breaker
        
        Args:
            initial_capital: Starting capital
            use_dynamic_sizing: True = compound, False = static
            leverage: 2.0 = 2x, 2.5 = 2.5x, 3.0 = 3x, etc.
            position_size_pct: Size per position (0.35 = 35%)
            max_positions: Maximum concurrent positions
            use_circuit_breaker: Enable circuit breaker protection
            circuit_breaker_threshold: Drawdown % that triggers breaker (e.g., -0.20 = -20%)
            circuit_breaker_mode: 'PAUSE' (stop new), 'REDUCE' (half size), 'STOP' (close all)
            circuit_breaker_reset: Drawdown % where trading resumes (e.g., -0.10 = -10%)
        """
        # Capital tracking
        self.initial_capital = initial_capital
        self.current_capital = initial_capital  # Cash available
        self.current_equity = initial_capital   # Total account value
        
        # Strategy configuration
        self.use_dynamic_sizing = use_dynamic_sizing
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        
        # Circuit Breaker configuration
        self.use_circuit_breaker = use_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_mode = circuit_breaker_mode
        self.circuit_breaker_reset = circuit_breaker_reset
        self.circuit_breaker_active = False
        self.circuit_breaker_events = []
        
        # Symbols (v49 winners)
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN']
        
        # Technical parameters
        self.rsi_period = 14
        self.rsi_buy_threshold = 55
        self.rsi_sell_threshold = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # Risk management
        self.stop_loss_pct = 0.08      # 8%
        self.take_profit_pct = 0.25     # 25%
        self.max_hold_days = 14
        
        # Tracking
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.data = {}
        
        # Statistics
        self.total_deployed = 0
        self.peak_equity = initial_capital
        self.max_drawdown = 0
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def load_data(self, start_date='2019-01-01', end_date=None):
        """Load price data for all symbols"""
        print(f"\n📊 Loading data for {len(self.symbols)} symbols...")
        if end_date:
            print(f"   Date Range: {start_date} to {end_date}")
        
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if len(df) > 100:
                    # Calculate indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    df = df.dropna()
                    
                    self.data[symbol] = df
                    print(f"  ✓ {symbol}: {len(df)} days")
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {e}")
        
        print(f"\n✓ Loaded {len(self.data)} symbols successfully\n")
    
    def check_circuit_breaker(self, current_date):
        """
        Check if circuit breaker should trigger or reset
        Returns True if trading is allowed, False if blocked
        """
        if not self.use_circuit_breaker:
            return True
        
        # Calculate current drawdown
        current_dd = (self.current_equity - self.peak_equity) / self.peak_equity
        
        # Check if we should trigger the breaker
        if not self.circuit_breaker_active and current_dd <= self.circuit_breaker_threshold:
            self.circuit_breaker_active = True
            event = {
                'date': current_date,
                'action': 'TRIGGERED',
                'drawdown': current_dd,
                'equity': self.current_equity,
                'mode': self.circuit_breaker_mode
            }
            self.circuit_breaker_events.append(event)
            print(f"\n🚨 CIRCUIT BREAKER TRIGGERED! 🚨")
            print(f"   Date: {current_date}")
            print(f"   Drawdown: {current_dd*100:.1f}%")
            print(f"   Mode: {self.circuit_breaker_mode}")
            
            # If STOP mode, close all positions immediately
            if self.circuit_breaker_mode == 'STOP':
                print(f"   Closing all positions...")
                for symbol in list(self.positions.keys()):
                    self.exit_trade(symbol, current_date, "CIRCUIT_BREAKER_STOP")
        
        # Check if we should reset the breaker
        elif self.circuit_breaker_active and current_dd >= self.circuit_breaker_reset:
            self.circuit_breaker_active = False
            event = {
                'date': current_date,
                'action': 'RESET',
                'drawdown': current_dd,
                'equity': self.current_equity,
                'mode': self.circuit_breaker_mode
            }
            self.circuit_breaker_events.append(event)
            print(f"\n✅ CIRCUIT BREAKER RESET")
            print(f"   Date: {current_date}")
            print(f"   Drawdown: {current_dd*100:.1f}%")
            print(f"   Trading resumed")
        
        # Return whether trading is allowed
        if self.circuit_breaker_active:
            if self.circuit_breaker_mode in ['PAUSE', 'STOP']:
                return False  # Block new trades
            else:  # REDUCE mode
                return True   # Allow trades but with reduced size
        
        return True  # Circuit breaker not active, allow normal trading
    
    def update_current_equity(self, current_date):
        """
        CRITICAL: Calculate total account equity
        This is what enables dynamic position sizing!
        """
        # Start with cash
        equity = self.current_capital
        
        # Add value of all open positions
        for symbol, pos in self.positions.items():
            if symbol in self.data:
                if current_date in self.data[symbol].index:
                    current_price = self.data[symbol].loc[current_date, 'Close']
                    
                    # Convert to scalar if needed (in case of duplicate dates)
                    if isinstance(current_price, pd.Series):
                        current_price = current_price.iloc[0]
                    
                    # Skip if price is invalid
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    position_value = pos['shares'] * current_price
                    equity += position_value
        
        self.current_equity = equity
        
        # Track for drawdown calculation
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        current_dd = (equity - self.peak_equity) / self.peak_equity
        if current_dd < self.max_drawdown:
            self.max_drawdown = current_dd
        
        return equity
    
    def calculate_position_size(self, symbol, price, current_date):
        """
        Calculate position size - with circuit breaker adjustment
        
        Static mode:  Uses initial_capital (fixed sizing)
        Dynamic mode: Uses current_equity (compounding)
        
        Circuit Breaker REDUCE mode: Cuts position size by 50%
        """
        # Update equity first
        self.update_current_equity(current_date)
        
        # Choose capital base
        if self.use_dynamic_sizing:
            base_capital = self.current_equity  # ← COMPOUNDING!
            mode = "DYNAMIC"
        else:
            base_capital = self.initial_capital  # ← STATIC
            mode = "STATIC"
        
        # Calculate position
        position_value = base_capital * self.position_size_pct * self.leverage
        
        # Apply circuit breaker reduction if active and in REDUCE mode
        if self.use_circuit_breaker and self.circuit_breaker_active and self.circuit_breaker_mode == 'REDUCE':
            position_value *= 0.5  # Cut in half
            mode += " (CB-REDUCED)"
        
        shares = int(position_value / price)
        cost = shares * price
        
        # Log for transparency
        if shares > 0:
            deployment_pct = (cost / self.current_equity) * 100
            print(f"    💰 {mode} Sizing: Equity=${self.current_equity:,.0f} → "
                  f"Position=${cost:,.0f} ({deployment_pct:.0f}%) = {shares} shares")
        
        return shares, cost
    
    def check_buy_signal(self, symbol, current_date):
        """V49 momentum buy signal (TREND_FOLLOW + PULLBACK)"""
        df = self.data[symbol]
        
        if current_date not in df.index:
            return False, 0
        
        # Extract scalar values directly
        price = df.loc[current_date, 'Close']
        rsi = df.loc[current_date, 'RSI']
        ma_fast = df.loc[current_date, 'MA_Fast']
        ma_slow = df.loc[current_date, 'MA_Slow']
        
        # Convert to scalar if needed (in case of duplicate dates)
        if isinstance(price, pd.Series):
            price = price.iloc[0]
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[0]
        if isinstance(ma_fast, pd.Series):
            ma_fast = ma_fast.iloc[0]
        if isinstance(ma_slow, pd.Series):
            ma_slow = ma_slow.iloc[0]
        
        # TREND_FOLLOW: Strong momentum
        if (rsi > 60 and 
            price > ma_fast > ma_slow):
            return True, 3  # Quality 3
        
        # PULLBACK: Dip in uptrend
        if (45 <= rsi <= 55 and 
            price > ma_slow and 
            ma_fast > ma_slow):
            return True, 2  # Quality 2
        
        return False, 0
    
    def check_exit_signal(self, symbol, position, current_date):
        """Check if position should be exited"""
        df = self.data[symbol]
        
        if current_date not in df.index:
            return False, "N/A"
        
        # Extract scalar values directly
        current_price = df.loc[current_date, 'Close']
        rsi = df.loc[current_date, 'RSI']
        
        # Convert to scalar if needed (in case of duplicate dates)
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[0]
        
        entry_price = position['entry_price']
        days_held = (current_date - position['entry_date']).days
        
        # Calculate return
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "STOP_LOSS"
        
        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return True, "TAKE_PROFIT"
        
        # Time exit
        if days_held >= self.max_hold_days:
            return True, "TIME_EXIT"
        
        # RSI sell signal
        if rsi < self.rsi_sell_threshold:
            return True, "RSI_SELL"
        
        return False, "HOLD"
    
    def enter_trade(self, symbol, current_date, quality):
        """Enter a new position with dynamic sizing"""
        df = self.data[symbol]
        
        if current_date not in df.index:
            return False
        
        price = df.loc[current_date, 'Close']
        
        # Convert to scalar if needed (in case of duplicate dates)
        if isinstance(price, pd.Series):
            price = price.iloc[0]
        
        # Validate price
        if pd.isna(price) or price <= 0:
            return False
        
        # Calculate position size (DYNAMIC OR STATIC)
        shares, cost = self.calculate_position_size(symbol, price, current_date)
        
        if shares == 0 or cost > self.current_capital:
            return False
        
        # Execute trade
        self.current_capital -= cost
        
        self.positions[symbol] = {
            'entry_date': current_date,
            'entry_price': price,
            'shares': shares,
            'cost': cost,
            'quality': quality
        }
        
        self.total_deployed += cost
        
        print(f"🟢 BUY {symbol}: {shares} shares @ ${price:.2f} "
              f"[Quality:{quality}/3] Cost=${cost:,.0f}")
        
        return True
    
    def exit_trade(self, symbol, current_date, reason):
        """Exit a position"""
        if symbol not in self.positions:
            return
        
        df = self.data[symbol]
        if current_date not in df.index:
            return
        
        position = self.positions[symbol]
        exit_price = df.loc[current_date, 'Close']
        
        # Convert to scalar if needed (in case of duplicate dates)
        if isinstance(exit_price, pd.Series):
            exit_price = exit_price.iloc[0]
        
        # Validate price
        if pd.isna(exit_price) or exit_price <= 0:
            return
        
        # Calculate P&L
        proceeds = position['shares'] * exit_price
        pnl = proceeds - position['cost']
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
        days_held = (current_date - position['entry_date']).days
        
        # Update capital
        self.current_capital += proceeds
        
        # Record trade
        trade = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': current_date,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': position['shares'],
            'cost': position['cost'],
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'reason': reason,
            'quality': position['quality']
        }
        
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        # Log
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} SELL {symbol}: ${pnl:,.0f} ({pnl_pct:+.1f}%) | {reason} | {days_held}d")
    
    def run_backtest(self):
        """Run the backtest with dynamic or static position sizing and circuit breaker"""
        print("\n" + "="*80)
        mode = "DYNAMIC" if self.use_dynamic_sizing else "STATIC"
        cb_status = f" + CIRCUIT BREAKER ({self.circuit_breaker_mode})" if self.use_circuit_breaker else ""
        print(f"V49 {mode} POSITION SIZING BACKTEST{cb_status}")
        print("="*80)
        print(f"Initial Capital:     ${self.initial_capital:,}")
        print(f"Position Sizing:     {mode}")
        print(f"Leverage:            {self.leverage}x")
        print(f"Position Size:       {self.position_size_pct*100:.0f}% per trade")
        print(f"Max Positions:       {self.max_positions}")
        print(f"Total Deployment:    {self.max_positions * self.position_size_pct * self.leverage * 100:.0f}%")
        
        if self.use_circuit_breaker:
            print(f"\n🚨 Circuit Breaker:  ENABLED")
            print(f"   Trigger:          {self.circuit_breaker_threshold*100:.0f}% drawdown")
            print(f"   Mode:             {self.circuit_breaker_mode}")
            print(f"   Reset:            {self.circuit_breaker_reset*100:.0f}% drawdown")
        
        print("="*80 + "\n")
        
        # Get all unique dates
        all_dates = sorted(set().union(*[set(df.index) for df in self.data.values()]))
        
        trade_count = 0
        
        for i, current_date in enumerate(all_dates):
            # Update equity and track
            equity = self.update_current_equity(current_date)
            
            # Check circuit breaker status
            trading_allowed = self.check_circuit_breaker(current_date)
            
            self.equity_history.append({
                'date': current_date,
                'equity': equity,
                'cash': self.current_capital,
                'positions': len(self.positions),
                'circuit_breaker': self.circuit_breaker_active
            })
            
            # Progress
            if i % 100 == 0:
                cb_indicator = " 🚨 CB ACTIVE" if self.circuit_breaker_active else ""
                print(f"Day {i}/{len(all_dates)} | Equity: ${equity:,.0f} | "
                      f"Positions: {len(self.positions)}/{self.max_positions} | Trades: {trade_count}{cb_indicator}")
            
            # Check exits first
            for symbol in list(self.positions.keys()):
                should_exit, reason = self.check_exit_signal(symbol, self.positions[symbol], current_date)
                if should_exit:
                    self.exit_trade(symbol, current_date, reason)
                    trade_count += 1
            
            # Check entries if we have room and trading is allowed
            if len(self.positions) < self.max_positions and trading_allowed:
                # Scan for signals
                signals = []
                for symbol in self.symbols:
                    if symbol not in self.positions and symbol in self.data:
                        has_signal, quality = self.check_buy_signal(symbol, current_date)
                        if has_signal:
                            signals.append((symbol, quality))
                
                # Sort by quality and enter best signals
                signals.sort(key=lambda x: x[1], reverse=True)
                
                for symbol, quality in signals:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    if self.enter_trade(symbol, current_date, quality):
                        trade_count += 1
        
        # Close any remaining positions
        final_date = all_dates[-1]
        for symbol in list(self.positions.keys()):
            self.exit_trade(symbol, final_date, "BACKTEST_END")
        
        # Final equity update
        final_equity = self.update_current_equity(final_date)
        
        # Print results
        self.print_results()
        
        # Save CSVs
        self.save_results()
    
    def print_results(self):
        """Print backtest results"""
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        
        # Basic metrics
        final_equity = self.current_equity
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        
        print(f"Initial Capital:     ${self.initial_capital:,}")
        print(f"Final Equity:        ${final_equity:,.0f}")
        print(f"Total Return:        {total_return:.1f}%")
        print(f"Max Drawdown:        {self.max_drawdown*100:.1f}%")
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            winners = trades_df[trades_df['pnl'] > 0]
            losers = trades_df[trades_df['pnl'] <= 0]
            
            win_rate = len(winners) / len(trades_df) * 100
            avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
            avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
            
            start_date = trades_df['entry_date'].min()
            end_date = trades_df['exit_date'].max()
            years = (end_date - start_date).days / 365.25
            cagr = (((final_equity / self.initial_capital) ** (1 / years)) - 1) * 100
            
            print(f"\nCAGR:                {cagr:.1f}%")
            print(f"Total Trades:        {len(trades_df)}")
            print(f"Win Rate:            {win_rate:.1f}%")
            print(f"Avg Win:             ${avg_win:,.0f}")
            print(f"Avg Loss:            ${avg_loss:,.0f}")
            
            if avg_loss != 0:
                profit_factor = abs(winners['pnl'].sum() / losers['pnl'].sum())
                print(f"Profit Factor:       {profit_factor:.2f}")
        
        # Circuit breaker statistics
        if self.use_circuit_breaker and len(self.circuit_breaker_events) > 0:
            print(f"\n🚨 Circuit Breaker Events: {len(self.circuit_breaker_events)}")
            triggers = [e for e in self.circuit_breaker_events if e['action'] == 'TRIGGERED']
            resets = [e for e in self.circuit_breaker_events if e['action'] == 'RESET']
            print(f"   Triggered:        {len(triggers)} times")
            print(f"   Reset:            {len(resets)} times")
            
            if len(triggers) > 0:
                print(f"\n   Events:")
                for event in self.circuit_breaker_events[:5]:  # Show first 5
                    print(f"   {event['date'].date()} | {event['action']:10} | DD: {event['drawdown']*100:6.1f}%")
        
        print("="*80)
    
    def save_results(self):
        """Save results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        mode = "dynamic" if self.use_dynamic_sizing else "static"
        cb_suffix = "_CB" if self.use_circuit_breaker else ""
        
        # Save trades
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            filename = f'v49_{mode}{cb_suffix}_trades_{timestamp}.csv'
            trades_df.to_csv(filename, index=False)
            print(f"\n📁 Trades saved: {filename}")
        
        # Save equity curve
        equity_df = pd.DataFrame(self.equity_history)
        if len(equity_df) > 0:
            filename = f'v49_{mode}{cb_suffix}_equity_{timestamp}.csv'
            equity_df.to_csv(filename, index=False)
            print(f"📁 Equity saved: {filename}")
        
        # Save circuit breaker events
        if self.use_circuit_breaker and len(self.circuit_breaker_events) > 0:
            cb_df = pd.DataFrame(self.circuit_breaker_events)
            filename = f'v49_{mode}{cb_suffix}_events_{timestamp}.csv'
            cb_df.to_csv(filename, index=False)
            print(f"📁 Circuit Breaker events saved: {filename}")


def compare_with_without_cb(initial_capital=20000, leverage=2.0, start_date='2019-01-01', end_date=None):
    """
    Compare performance with and without circuit breaker
    """
    print("\n" + "╔" + "="*86 + "╗")
    print("║" + " "*86 + "║")
    print("║" + " "*22 + "CIRCUIT BREAKER COMPARISON" + " "*38 + "║")
    print("║" + " "*86 + "║")
    print("╚" + "="*86 + "╝\n")
    
    results = []
    
    # Test configurations
    configs = [
        {'name': 'Dynamic 2x - NO CB', 'use_cb': False, 'threshold': -0.20, 'mode': 'PAUSE'},
        {'name': 'Dynamic 2x - CB PAUSE -20%', 'use_cb': True, 'threshold': -0.20, 'mode': 'PAUSE'},
        {'name': 'Dynamic 2x - CB PAUSE -15%', 'use_cb': True, 'threshold': -0.15, 'mode': 'PAUSE'},
        {'name': 'Dynamic 2x - CB REDUCE -20%', 'use_cb': True, 'threshold': -0.20, 'mode': 'REDUCE'},
        {'name': 'Dynamic 2x - CB STOP -25%', 'use_cb': True, 'threshold': -0.25, 'mode': 'STOP'},
    ]
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        
        strategy = V49WithCircuitBreaker(
            initial_capital=initial_capital,
            use_dynamic_sizing=True,
            leverage=leverage,
            position_size_pct=0.35,
            max_positions=3,
            use_circuit_breaker=config['use_cb'],
            circuit_breaker_threshold=config['threshold'],
            circuit_breaker_mode=config['mode'],
            circuit_breaker_reset=config['threshold'] / 2  # Reset at half the trigger
        )
        
        strategy.load_data(start_date=start_date, end_date=end_date)
        strategy.run_backtest()
        
        # Extract metrics
        final_equity = strategy.current_equity
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        trades_df = pd.DataFrame(strategy.trades)
        if len(trades_df) > 0:
            start = trades_df['entry_date'].min()
            end = trades_df['exit_date'].max()
            years = (end - start).days / 365.25
            cagr = (((final_equity / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
        else:
            cagr = 0
        
        cb_events = len(strategy.circuit_breaker_events)
        
        results.append({
            'Configuration': config['name'],
            'CAGR': f"{cagr:.1f}%",
            'MaxDD': f"{strategy.max_drawdown*100:.1f}%",
            'Final': f"${final_equity:,.0f}",
            'Trades': len(trades_df),
            'CB Events': cb_events if config['use_cb'] else 'N/A'
        })
    
    # Print comparison
    print("\n" + "="*100)
    print("CIRCUIT BREAKER COMPARISON")
    print("="*100)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("="*100)
    print("\n💡 ANALYSIS:")
    print("  • Compare MaxDD columns - did circuit breaker reduce drawdown?")
    print("  • Compare CAGR - how much return did you sacrifice?")
    print("  • Check CB Events - how often was it triggered?")
    print("  • Best config: Lower MaxDD with minimal CAGR sacrifice")
    print("="*100)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare-cb':
            # Compare with/without circuit breaker
            compare_with_without_cb(
                initial_capital=20000,
                leverage=2.0,
                start_date='2019-01-01'
            )
        elif sys.argv[1] == '--ytd2025-cb':
            # YTD 2025 comparison with circuit breaker
            compare_with_without_cb(
                initial_capital=20000,
                leverage=2.0,
                start_date='2024-12-01',
                end_date='2025-10-18'
            )
        else:
            print("Usage:")
            print("  python v49_circuit_breaker.py                # Single backtest")
            print("  python v49_circuit_breaker.py --compare-cb   # Compare with/without CB")
            print("  python v49_circuit_breaker.py --ytd2025-cb   # YTD 2025 CB comparison")
    else:
        # Run single backtest with circuit breaker
        strategy = V49WithCircuitBreaker(
            initial_capital=20000,
            use_dynamic_sizing=True,
            leverage=2.0,
            position_size_pct=0.35,
            max_positions=3,
            use_circuit_breaker=True,        # ← ENABLE/DISABLE HERE
            circuit_breaker_threshold=-0.20,  # -20% drawdown triggers
            circuit_breaker_mode='PAUSE',     # PAUSE, REDUCE, or STOP
            circuit_breaker_reset=-0.10       # Reset at -10%
        )
        
        strategy.load_data(start_date='2019-01-01')
        strategy.run_backtest()
        
        print("\n✅ Complete!")
        print("\n💡 To compare with/without circuit breaker:")
        print("   python v49_circuit_breaker.py --compare-cb")
