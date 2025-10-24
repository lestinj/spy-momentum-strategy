"""
V48 REGIME FILTER
=================

Testing: Can we reduce drawdowns by detecting deteriorating market regimes
and sitting out trades during bad periods?

Concept: Use "second derivative" thinking - trends in indicators rather than
static thresholds to detect when conditions are getting worse.

Regime Indicators:
1. Market RSI Momentum - Is the average RSI across stocks trending down?
2. Failed Bounce Rate - Are recent RSI ≤ 30 signals failing to bounce?
3. Equity Momentum - Is our equity curve accelerating downward?
4. Volatility Expansion - Is market volatility increasing?

If multiple indicators show deterioration → PAUSE trading
If indicators show improvement → RESUME trading

This preserves V48's core while adding an adaptive filter based on market regime,
not individual trade quality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class V48RegimeFilter:
    """
    V48 with Regime Detection
    
    Same core strategy as V48, but with ability to pause trading
    when market regime deteriorates.
    """
    
    def __init__(self, initial_capital=30000):
        self.initial_capital = initial_capital
        self.max_positions = 3
        self.position_size_pct = 1.0 / self.max_positions
        
        # V48 Core Parameters
        self.rsi_entry = 30
        self.rsi_exit = 70
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.15
        
        # Regime Detection Parameters
        self.regime_lookback = 20  # Days to analyze for regime
        self.failed_bounce_threshold = 0.6  # 60% of recent signals failing = bad regime
        self.equity_momentum_threshold = -0.05  # -5% over lookback = pause
        self.volatility_expansion_threshold = 1.5  # 50% above average = pause
        self.min_regime_indicators = 2  # How many must trigger to pause
        
        # Tracking
        self.regime_pauses = 0
        self.trades_skipped = 0

        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']

        self.data = None
    
    def load_data(self, start_date='2024-01-01', end_date=None):
        """Load market data"""
        print("📊 Loading market data...")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            import yfinance as yf
            self.data = yf.download(
                self.symbols,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False
            )
            print(f"✅ Data loaded: {start_date} to {end_date}")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data(self):
        """Prepare data with indicators"""
        all_data = {}
        
        for symbol in self.symbols:
            df = self.data[symbol].copy()
            df['RSI'] = self.calculate_rsi(df['Close'])
            
            # Add volatility measure (ATR as % of price)
            df['High_Low'] = df['High'] - df['Low']
            df['ATR'] = df['High_Low'].rolling(window=14).mean()
            df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
            
            all_data[symbol] = df
        
        return all_data
    
    def detect_regime(self, date, all_data, equity_curve, recent_signals):
        """
        Detect if market regime is deteriorating
        Returns: (is_good_regime, regime_score, reasons)
        """
        if len(equity_curve) < self.regime_lookback:
            return True, 1.0, ["Insufficient data"]
        
        warning_signals = []
        warning_count = 0
        
        # 1. EQUITY MOMENTUM - Is our equity accelerating downward?
        recent_equity = [e['equity'] for e in equity_curve[-self.regime_lookback:]]
        equity_change = (recent_equity[-1] - recent_equity[0]) / recent_equity[0]
        
        if equity_change < self.equity_momentum_threshold:
            warning_signals.append(f"Equity momentum: {equity_change*100:.1f}%")
            warning_count += 1
        
        # 2. FAILED BOUNCE RATE - Are RSI signals failing to bounce?
        if len(recent_signals) >= 5:  # Need at least 5 recent signals
            failed = sum(1 for s in recent_signals[-10:] if s['failed'])
            total = len(recent_signals[-10:])
            failed_rate = failed / total
            
            if failed_rate > self.failed_bounce_threshold:
                warning_signals.append(f"Failed bounces: {failed_rate*100:.0f}%")
                warning_count += 1
        
        # 3. MARKET RSI TREND - Is average RSI trending lower?
        try:
            avg_rsi_now = np.mean([all_data[sym].loc[date, 'RSI'] 
                                   for sym in self.symbols 
                                   if date in all_data[sym].index])
            
            date_idx = all_data[self.symbols[0]].index.get_loc(date)
            if date_idx >= self.regime_lookback:
                past_date = all_data[self.symbols[0]].index[date_idx - self.regime_lookback]
                avg_rsi_past = np.mean([all_data[sym].loc[past_date, 'RSI'] 
                                       for sym in self.symbols 
                                       if past_date in all_data[sym].index])
                
                rsi_momentum = avg_rsi_now - avg_rsi_past
                
                if rsi_momentum < -10:  # Market RSI dropped 10+ points
                    warning_signals.append(f"Market RSI momentum: {rsi_momentum:.1f}")
                    warning_count += 1
        except:
            pass
        
        # 4. VOLATILITY EXPANSION - Is market volatility spiking?
        try:
            current_vol = np.mean([all_data[sym].loc[date, 'ATR_Pct'] 
                                  for sym in self.symbols 
                                  if date in all_data[sym].index])
            
            # Compare to lookback average
            vol_history = []
            for i in range(self.regime_lookback):
                if date_idx - i >= 0:
                    past_date = all_data[self.symbols[0]].index[date_idx - i]
                    past_vol = np.mean([all_data[sym].loc[past_date, 'ATR_Pct'] 
                                       for sym in self.symbols 
                                       if past_date in all_data[sym].index])
                    vol_history.append(past_vol)
            
            avg_vol = np.mean(vol_history)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            if vol_ratio > self.volatility_expansion_threshold:
                warning_signals.append(f"Volatility spike: {vol_ratio:.2f}x")
                warning_count += 1
        except:
            pass
        
        # DECISION: Pause if enough warning signals
        is_good_regime = warning_count < self.min_regime_indicators
        regime_score = 1.0 - (warning_count / 4.0)  # 4 possible warnings
        
        return is_good_regime, regime_score, warning_signals
    
    def run_backtest(self):
        """Execute backtest with regime filtering"""
        print("\n🚀 Running V48 Regime Filter Backtest...")
        print("=" * 80)
        
        all_data = self.prepare_data()
        
        all_dates = set(all_data[self.symbols[0]].index)
        for symbol in self.symbols[1:]:
            all_dates &= set(all_data[symbol].index)
        all_dates = sorted(list(all_dates))
        
        # Initialize
        equity = self.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        recent_signals = []  # Track recent signal outcomes
        regime_history = []
        
        print(f"📅 Period: {all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.0f}")
        print(f"🎯 Max Positions: {self.max_positions}")
        print(f"🔍 Regime Detection: ENABLED")
        print()
        
        # Main loop
        for date in all_dates:
            # Check exits
            positions_to_close = []
            
            for symbol, pos in positions.items():
                current_price = all_data[symbol].loc[date, 'Close']
                current_rsi = all_data[symbol].loc[date, 'RSI']
                
                exit_reason = None
                if current_price <= pos['stop']:
                    exit_reason = 'Stop Loss'
                elif current_price >= pos['target']:
                    exit_reason = 'Take Profit'
                elif current_rsi >= self.rsi_exit:
                    exit_reason = 'RSI Exit'
                
                if exit_reason:
                    exit_value = pos['shares'] * current_price
                    pnl = exit_value - (pos['shares'] * pos['entry_price'])
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    
                    equity += pnl
                    
                    # Track if signal was successful (for regime detection)
                    signal_success = pnl > 0
                    if 'signal_id' in pos:
                        for sig in recent_signals:
                            if sig['id'] == pos['signal_id']:
                                sig['failed'] = not signal_success
                                break
                    
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
            
            for symbol in positions_to_close:
                del positions[symbol]
            
            # Record equity before checking regime
            positions_value = sum(
                pos['shares'] * all_data[symbol].loc[date, 'Close']
                for symbol, pos in positions.items()
            )
            cash = equity - sum(pos['shares'] * pos['entry_price'] for pos in positions.values())
            total_equity = cash + positions_value
            
            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'positions': len(positions),
                'cash': cash
            })
            
            # REGIME CHECK - Should we take new trades?
            is_good_regime, regime_score, regime_reasons = self.detect_regime(
                date, all_data, equity_curve, recent_signals
            )
            
            regime_history.append({
                'date': date,
                'is_good': is_good_regime,
                'score': regime_score,
                'reasons': regime_reasons
            })
            
            # Look for new entries (only if regime is good)
            if len(positions) < self.max_positions:
                entry_candidates = []
                
                for symbol in self.symbols:
                    if symbol in positions:
                        continue
                    
                    current_price = all_data[symbol].loc[date, 'Close']
                    current_rsi = all_data[symbol].loc[date, 'RSI']
                    
                    if current_rsi <= self.rsi_entry:
                        entry_candidates.append({
                            'symbol': symbol,
                            'price': current_price,
                            'rsi': current_rsi
                        })
                
                if entry_candidates:
                    if is_good_regime:
                        # TAKE THE TRADES - regime is good
                        entry_candidates.sort(key=lambda x: x['rsi'])
                        slots_available = self.max_positions - len(positions)
                        
                        for candidate in entry_candidates[:slots_available]:
                            symbol = candidate['symbol']
                            entry_price = candidate['price']
                            
                            position_value = equity * self.position_size_pct
                            shares = int(position_value / entry_price)
                            
                            if shares > 0:
                                signal_id = f"{symbol}_{date.strftime('%Y%m%d')}"
                                
                                positions[symbol] = {
                                    'entry_price': entry_price,
                                    'entry_date': date,
                                    'shares': shares,
                                    'stop': entry_price * (1 - self.stop_loss_pct),
                                    'target': entry_price * (1 + self.take_profit_pct),
                                    'signal_id': signal_id
                                }
                                
                                recent_signals.append({
                                    'id': signal_id,
                                    'date': date,
                                    'symbol': symbol,
                                    'failed': False  # Will update on exit
                                })
                                
                                # Keep only recent signals
                                if len(recent_signals) > 20:
                                    recent_signals.pop(0)
                    else:
                        # SKIP THE TRADES - regime is bad
                        self.trades_skipped += len(entry_candidates)
                        if regime_history[-1]['is_good'] != is_good_regime:
                            self.regime_pauses += 1
        
        # Close remaining positions
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
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        regime_df = pd.DataFrame(regime_history)
        
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
        win_rate = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # Regime statistics
        regime_pauses_pct = (regime_df['is_good'] == False).sum() / len(regime_df) * 100
        
        # Print results
        print("\n" + "="*80)
        print("V48 REGIME FILTER RESULTS")
        print("="*80)
        print(f"Initial Capital:    ${self.initial_capital:>15,.0f}")
        print(f"Final Equity:       ${final_equity:>15,.0f}")
        print(f"Total Return:       {total_return:>15.1f}%")
        print(f"CAGR:               {cagr:>15.1f}%")
        print(f"Max Drawdown:       {max_dd:>15.1f}%")
        print(f"\nTotal Trades:       {len(trades_df):>15,}")
        print(f"Winners:            {len(winning):>15,} ({win_rate:.1f}%)")
        print(f"Losers:             {len(losing):>15,}")
        
        if len(winning) > 0:
            print(f"Avg Win:            ${winning['pnl'].mean():>14,.0f}")
        if len(losing) > 0:
            print(f"Avg Loss:           ${losing['pnl'].mean():>14,.0f}")
            pf = abs(winning['pnl'].sum() / losing['pnl'].sum())
            print(f"Profit Factor:      {pf:>15.2f}")
        
        print("\n" + "="*80)
        print("REGIME DETECTION IMPACT")
        print("="*80)
        print(f"Trading Days:       {len(regime_df):>15,}")
        print(f"Good Regime Days:   {(regime_df['is_good']).sum():>15,} ({100-regime_pauses_pct:.1f}%)")
        print(f"Bad Regime Days:    {(~regime_df['is_good']).sum():>15,} ({regime_pauses_pct:.1f}%)")
        print(f"Regime Transitions: {self.regime_pauses:>15,}")
        print(f"Trades Skipped:     {self.trades_skipped:>15,}")
        
        print("\n" + "="*80)
        print("💡 COMPARISON VS V48 BASELINE")
        print("="*80)
        print(f"V48 Baseline:       195.0% CAGR, ~25% max DD")
        print(f"V48 Regime Filter:  {cagr:>5.1f}% CAGR, {max_dd:.1f}% max DD")
        print(f"\nCAGR Difference:    {cagr - 195:+.1f}%")
        print(f"Drawdown Change:    {max_dd - (-25):+.1f}%")
        
        if cagr > 195 and max_dd > -25:
            print("\n✅ SUCCESS: Higher returns AND lower drawdown!")
        elif cagr > 195:
            print("\n⚠️  Higher returns but similar/worse drawdown")
        elif max_dd > -25:
            print("\n⚠️  Lower drawdown but at cost of returns")
        else:
            print("\n❌ Regime filter hurt both returns and drawdowns")
        
        # Save files
        trades_df.to_csv('v48_regime_trades.csv', index=False)
        equity_df.to_csv('v48_regime_equity.csv', index=False)
        regime_df.to_csv('v48_regime_history.csv', index=False)
        
        print(f"\n📁 Files saved:")
        print(f"  • v48_regime_trades.csv")
        print(f"  • v48_regime_equity.csv")
        print(f"  • v48_regime_history.csv (regime detection log)")
        
        print("\n" + "="*80)
        
        return {
            'trades': trades_df,
            'equity': equity_df,
            'regime': regime_df,
            'final_equity': final_equity,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'regime_pauses_pct': regime_pauses_pct
        }


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                    V48 REGIME FILTER                                  ║
    ║                                                                        ║
    ║  Testing: Can regime detection reduce drawdowns?                      ║
    ║                                                                        ║
    ║  Approach:                                                            ║
    ║    • Keep V48's core strategy intact                                  ║
    ║    • Add regime detection layer                                       ║
    ║    • Pause trading when conditions deteriorate                        ║
    ║    • Resume when conditions improve                                   ║
    ║                                                                        ║
    ║  Regime Indicators:                                                   ║
    ║    1. Equity momentum (is our account trending down?)                 ║
    ║    2. Failed bounce rate (are RSI signals failing?)                   ║
    ║    3. Market RSI trend (is average RSI falling?)                      ║
    ║    4. Volatility expansion (is market vol spiking?)                   ║
    ║                                                                        ║
    ║  Goal: Reduce drawdowns while maintaining returns                     ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    strategy = V48RegimeFilter(initial_capital=10000)
    strategy.load_data(start_date='2025-01-01')
    results = strategy.run_backtest()
    
    print("\n✅ Backtest Complete!")
    print("\nReview the comparison vs V48 baseline above.")
    print("Check v48_regime_history.csv to see when regime pauses occurred.")
