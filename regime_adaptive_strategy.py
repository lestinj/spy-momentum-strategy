"""
Regime-Adaptive Strategy V5.2 - STANDALONE VERSION
No external dependencies - complete in one file

Based on V5 actual results showing:
- 303 stop losses with 0% win rate → Fixed with 2.5% stops
- SQQQ failed (11.1% win rate) → Removed
- VXX worked (56.2% win rate) → Kept as only bear symbol
- 2021 only 9.5% return → Fixed with better position sizing

This version includes all code inline - no imports needed.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class RegimeAdaptiveV52:
    """Standalone V5.2 - Dual regime strategy with data-driven fixes"""
    
    def __init__(
        self,
        bull_symbols: List[str] = None,
        bear_symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
    ):
        # Symbol sets
        self.bull_symbols = bull_symbols or ["NVDA", "TSLA", "PLTR", "AMD", "COIN"]
        self.bear_symbols = bear_symbols or ["VXX"]  # Only VXX - it worked in V5
        
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # Technical indicators
        self.fast_ma = 10
        self.slow_ma = 30
        self.trend_ma = 50
        self.long_ma = 200
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        
        # === BULL MODE PARAMETERS (V5.2 FIXED) ===
        self.bull_base_risk = 0.025        # 2.5% (was 2.0%) - better participation
        self.bull_quality_mult = 1.5
        self.bull_momentum_mult = 2.0
        self.bull_stop_loss = 0.025        # 2.5% (was 1.5%) - WIDER STOPS
        self.bull_take_profit = 0.10       # 10% (was 8%) - let winners run
        self.bull_max_hold = 45
        self.bull_max_positions = 3        # 3 (was 5) - focus on quality
        
        # === BEAR MODE PARAMETERS (V5.2 OPTIMIZED FOR VXX) ===
        self.bear_base_risk = 0.020        # 2.0% - VXX doesn't decay
        self.bear_quality_mult = 1.3
        self.bear_stop_loss = 0.020        # 2.0% (was 1.0%)
        self.bear_take_profit = 0.15       # 15% - VXX can move big
        self.bear_max_hold = 60            # 60 days (VXX avg 56.6 in V5)
        self.bear_max_positions = 1        # Just VXX
        self.bear_fast_exit_days = 20      # Give VXX time
        
        # Signal quality
        self.min_signal_quality = 2        # Only trade quality ≥2
        
        # Regime detection
        self.regime_confirmation_days = 5
        self.current_regime = "BULL"
        self.regime_switch_pending = False
        self.regime_switch_date = None
        
        # Tracking
        self.options_multiplier = 100
        self.spy_data = None
        
        print("\n" + "="*80)
        print("REGIME-ADAPTIVE STRATEGY V5.2 - STANDALONE")
        print("Data-driven fixes from V5 analysis")
        print("="*80)
        print("\n🔧 KEY FIXES FROM V5:")
        print(f"  ✅ Bull stop loss: 1.5% → 2.5% (V5 had 303 stops with 0% win rate)")
        print(f"  ✅ Bear symbols: SQQQ,PG,KO,XLU,VXX → VXX only (only 56% winner)")
        print(f"  ✅ Bull positions: 5 → 3 (focus on quality)")
        print(f"  ✅ Position size: 2.0% → 2.5% (better participation)")
        print(f"  ✅ VXX max hold: 15d → 60d (V5 avg was 56.6 days)")
        print("="*80 + "\n")
    
    def load_spy_data(self) -> pd.DataFrame:
        """Enhanced SPY loading with bear detection"""
        print("Loading SPY for regime detection...")
        
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            extended_start = start - pd.Timedelta(days=300)
            
            spy = yf.Ticker("SPY")
            df = spy.history(start=extended_start, end=end, interval='1d')
            
            if df.empty:
                print("  ✗ Failed to load SPY")
                return None
            
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Bear detection indicators
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            df['below_200ma'] = df['close'] < df['sma_200']
            df['distance_from_200'] = (df['close'] - df['sma_200']) / df['sma_200']
            df['death_cross'] = df['sma_50'] < df['sma_200']
            df['momentum_negative'] = df['close'].pct_change(20) < -0.05
            
            # Volatility
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252) * 100
            df['high_volatility'] = df['volatility_20'] > 25
            
            # Composite bear signal
            df['bear_signals'] = (
                df['below_200ma'].astype(int) +
                df['death_cross'].astype(int) +
                df['momentum_negative'].astype(int) +
                df['high_volatility'].astype(int)
            )
            df['is_bear_market'] = df['bear_signals'] >= 2
            df['severe_bear'] = df['is_bear_market'] & (df['distance_from_200'] < -0.05)
            
            df = df[start:end]
            
            bear_days = df['is_bear_market'].sum()
            bear_pct = bear_days / len(df) * 100
            
            print(f"  ✓ SPY loaded: {len(df)} days")
            print(f"  Bear days: {bear_days} ({bear_pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error loading SPY: {e}")
            return None
    
    def detect_regime(self, date: datetime) -> str:
        """Detect regime with confirmation period"""
        if self.spy_data is None or date not in self.spy_data.index:
            return self.current_regime
        
        spy_row = self.spy_data.loc[date]
        is_bear = spy_row.get('is_bear_market', False)
        new_regime = "BEAR" if is_bear else "BULL"
        
        # Require confirmation
        if new_regime != self.current_regime:
            if not self.regime_switch_pending:
                self.regime_switch_pending = True
                self.regime_switch_date = date
                return self.current_regime
            else:
                days_pending = (date - self.regime_switch_date).days
                if days_pending >= self.regime_confirmation_days:
                    print(f"\n🔄 REGIME SWITCH: {self.current_regime} → {new_regime} on {date.date()}")
                    self.current_regime = new_regime
                    self.regime_switch_pending = False
                    return new_regime
                return self.current_regime
        else:
            self.regime_switch_pending = False
            return self.current_regime
    
    def load_single_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data with indicators"""
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            extended_start = start - pd.Timedelta(days=250)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=extended_start, end=end, interval='1d')
            
            if df.empty:
                return None
            
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Moving averages
            df['sma_fast'] = df['close'].rolling(self.fast_ma).mean()
            df['sma_slow'] = df['close'].rolling(self.slow_ma).mean()
            df['sma_trend'] = df['close'].rolling(self.trend_ma).mean()
            df['sma_long'] = df['close'].rolling(self.long_ma).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(self.bb_period).mean()
            df['bb_std'] = df['close'].rolling(self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['high_volume'] = df['volume'] > df['volume_sma'] * 1.2
            
            # Momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            
            # Trends
            df['uptrend'] = (df['sma_fast'] > df['sma_slow']) & (df['sma_slow'] > df['sma_trend'])
            df['downtrend'] = (df['sma_fast'] < df['sma_slow']) & (df['sma_slow'] < df['sma_trend'])
            df['perfect_uptrend'] = df['uptrend'] & (df['close'] > df['sma_fast'])
            df['perfect_downtrend'] = df['downtrend'] & (df['close'] < df['sma_fast'])
            
            df = df[start:end]
            return df
            
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all symbols"""
        print(f"\nLoading data...")
        print(f"Bull symbols: {', '.join(self.bull_symbols)}")
        print(f"Bear symbols: {', '.join(self.bear_symbols)}")
        print(f"Period: {self.start_date} to {self.end_date}\n")
        
        # Load SPY
        self.spy_data = self.load_spy_data()
        
        all_data = {}
        all_symbols = list(set(self.bull_symbols + self.bear_symbols))
        
        for symbol in all_symbols:
            print(f"Loading {symbol}...", end=" ")
            df = self.load_single_symbol(symbol)
            if df is not None and len(df) > 0:
                # Add SPY regime data
                if self.spy_data is not None:
                    df = df.join(
                        self.spy_data[['is_bear_market', 'severe_bear', 'distance_from_200']], 
                        how='left'
                    )
                    df['spy_bear'] = df['is_bear_market'].fillna(False)
                else:
                    df['spy_bear'] = False
                
                all_data[symbol] = df
                print(f"✓ {len(df)} days")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Successfully loaded {len(all_data)} symbols\n")
        return all_data
    
    def generate_signals(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate bull and bear signals"""
        print("="*80)
        print("GENERATING SIGNALS")
        print("="*80)
        
        for symbol, df in all_data.items():
            # BULL SIGNALS
            trend_long = (
                df['uptrend'] &
                (df['rsi'] > 40) & (df['rsi'] < 70) &
                (df['momentum_10'] > 0)
            )
            
            breakout_long = (
                (df['close'] > df['bb_upper']) &
                df['high_volume'] &
                df['perfect_uptrend'] &
                (df['momentum_10'] > 0.05)
            )
            
            pullback_long = (
                df['perfect_uptrend'] &
                (df['close'] < df['sma_fast']) &
                (df['close'] > df['sma_slow']) &
                (df['rsi'] < 50)
            )
            
            df['bull_signal'] = 0
            df['bull_signal_type'] = 'none'
            df.loc[trend_long, ['bull_signal', 'bull_signal_type']] = [1, 'bull_trend']
            df.loc[breakout_long, ['bull_signal', 'bull_signal_type']] = [2, 'bull_breakout']
            df.loc[pullback_long, ['bull_signal', 'bull_signal_type']] = [3, 'bull_pullback']
            
            # Bull quality
            df['bull_quality'] = 0
            df.loc[df['bull_signal'] > 0, 'bull_quality'] = 1
            high_quality_bull = (
                (df['bull_signal'] > 0) &
                df['perfect_uptrend'] &
                df['high_volume'] &
                (df['momentum_10'] > 0.05)
            )
            df.loc[high_quality_bull, 'bull_quality'] = 2
            
            # BEAR SIGNALS
            trend_short = (
                df['downtrend'] &
                (df['rsi'] < 60) & (df['rsi'] > 30) &
                (df['momentum_10'] < 0)
            )
            
            breakdown_short = (
                (df['close'] < df['bb_lower']) &
                df['high_volume'] &
                df['perfect_downtrend'] &
                (df['momentum_10'] < -0.05)
            )
            
            bounce_short = (
                df['perfect_downtrend'] &
                (df['close'] > df['sma_fast']) &
                (df['close'] < df['sma_slow']) &
                (df['rsi'] > 50) & (df['rsi'] < 65)
            )
            
            df['bear_signal'] = 0
            df['bear_signal_type'] = 'none'
            df.loc[trend_short, ['bear_signal', 'bear_signal_type']] = [1, 'bear_trend']
            df.loc[breakdown_short, ['bear_signal', 'bear_signal_type']] = [2, 'bear_breakdown']
            df.loc[bounce_short, ['bear_signal', 'bear_signal_type']] = [3, 'bear_bounce']
            
            # Bear quality
            df['bear_quality'] = 0
            df.loc[df['bear_signal'] > 0, 'bear_quality'] = 1
            high_quality_bear = (
                (df['bear_signal'] > 0) &
                df['perfect_downtrend'] &
                df['high_volume'] &
                (df['momentum_10'] < -0.05)
            )
            df.loc[high_quality_bear, 'bear_quality'] = 2
            
            all_data[symbol] = df
            
            bull_sigs = (df['bull_signal'] > 0).sum()
            bear_sigs = (df['bear_signal'] > 0).sum()
            print(f"{symbol}: {bull_sigs} bull, {bear_sigs} bear signals")
        
        print()
        return all_data
    
    def backtest(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run backtest"""
        print("="*80)
        print("RUNNING BACKTEST")
        print("="*80 + "\n")
        
        trades = []
        positions = {}
        capital = self.initial_capital
        
        all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        equity_curve = []
        regime_history = []
        
        for current_date in all_dates:
            regime = self.detect_regime(current_date)
            regime_history.append({'date': current_date, 'regime': regime})
            
            active_symbols = self.bear_symbols if regime == "BEAR" else self.bull_symbols
            
            current_prices = {}
            for symbol in active_symbols:
                if symbol in all_data and current_date in all_data[symbol].index:
                    current_prices[symbol] = all_data[symbol].loc[current_date]
            
            # Manage positions
            symbols_to_remove = []
            
            for symbol, pos in list(positions.items()):
                if symbol not in current_prices:
                    continue
                
                row = current_prices[symbol]
                current_price = row['close']
                days_held = (current_date - pos['entry_date']).days
                profit_pct = (current_price - pos['entry_price']) / pos['entry_price']
                
                if pos['direction'] == 'SHORT':
                    profit_pct = -profit_pct
                
                # Parameters
                if pos['direction'] == 'LONG':
                    max_hold = self.bull_max_hold
                    stop_pct = self.bull_stop_loss
                    target_pct = self.bull_take_profit
                else:
                    max_hold = self.bear_max_hold
                    stop_pct = self.bear_stop_loss
                    target_pct = self.bear_take_profit
                    
                    if days_held >= self.bear_fast_exit_days and profit_pct < 0.05:
                        pnl = (current_price - pos['entry_price']) * pos['shares'] * self.options_multiplier
                        if pos['direction'] == 'SHORT':
                            pnl = -pnl
                        capital += pnl
                        
                        trades.append({
                            'symbol': symbol,
                            'direction': pos['direction'],
                            'entry_date': pos['entry_date'],
                            'exit_date': current_date,
                            'entry_price': pos['entry_price'],
                            'exit_price': current_price,
                            'shares': pos['shares'],
                            'pnl': pnl,
                            'return_pct': profit_pct * 100,
                            'exit_reason': 'Bear No Progress',
                            'days_held': days_held,
                            'regime': pos['regime'],
                            'quality': pos.get('quality', 0)
                        })
                        symbols_to_remove.append(symbol)
                        continue
                
                # Exit logic
                exit_trade = False
                exit_reason = ""
                
                if pos['direction'] == 'LONG':
                    if current_price <= pos['stop_loss']:
                        exit_trade, exit_reason = True, "Stop Loss"
                    elif current_price >= pos['take_profit']:
                        exit_trade, exit_reason = True, "Take Profit"
                    elif not row.get('uptrend', False) and profit_pct < -0.01:
                        exit_trade, exit_reason = True, "Trend Break"
                else:
                    if current_price >= pos['stop_loss']:
                        exit_trade, exit_reason = True, "Stop Loss"
                    elif current_price <= pos['take_profit']:
                        exit_trade, exit_reason = True, "Take Profit"
                    elif not row.get('downtrend', False) and profit_pct < -0.02:
                        exit_trade, exit_reason = True, "Trend Break"
                
                if days_held > max_hold:
                    exit_trade, exit_reason = True, "Time Exit"
                
                if (regime == "BULL" and pos['direction'] == "SHORT") or \
                   (regime == "BEAR" and pos['direction'] == "LONG"):
                    exit_trade, exit_reason = True, "Regime Mismatch"
                
                if exit_trade:
                    pnl = (current_price - pos['entry_price']) * pos['shares'] * self.options_multiplier
                    if pos['direction'] == 'SHORT':
                        pnl = -pnl
                    capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'direction': pos['direction'],
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'return_pct': profit_pct * 100,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'regime': pos['regime'],
                        'quality': pos.get('quality', 0)
                    })
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del positions[symbol]
            
            # New entries
            max_positions = self.bear_max_positions if regime == "BEAR" else self.bull_max_positions
            
            if len(positions) < max_positions:
                opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol in positions:
                        continue
                    
                    if regime == "BULL" and row.get('bull_signal', 0) > 0:
                        quality = row.get('bull_quality', 1)
                        if quality >= self.min_signal_quality:
                            opportunities.append({
                                'symbol': symbol,
                                'quality': quality,
                                'price': row['close'],
                                'direction': 'LONG',
                                'signal_type': row.get('bull_signal_type', 'unknown')
                            })
                    elif regime == "BEAR" and row.get('bear_signal', 0) > 0:
                        quality = row.get('bear_quality', 1)
                        if quality >= self.min_signal_quality:
                            opportunities.append({
                                'symbol': symbol,
                                'quality': quality,
                                'price': row['close'],
                                'direction': 'SHORT',
                                'signal_type': row.get('bear_signal_type', 'unknown')
                            })
                
                opportunities.sort(key=lambda x: x['quality'], reverse=True)
                
                for opp in opportunities[:max_positions - len(positions)]:
                    symbol = opp['symbol']
                    direction = opp['direction']
                    current_price = opp['price']
                    
                    if direction == 'LONG':
                        base_risk = self.bull_base_risk
                        if opp['quality'] == 2:
                            base_risk *= self.bull_quality_mult
                        stop_pct = self.bull_stop_loss
                        target_pct = self.bull_take_profit
                        stop_loss = current_price * (1 - stop_pct)
                        take_profit = current_price * (1 + target_pct)
                    else:
                        base_risk = self.bear_base_risk
                        if opp['quality'] == 2:
                            base_risk *= self.bear_quality_mult
                        stop_pct = self.bear_stop_loss
                        target_pct = self.bear_take_profit
                        stop_loss = current_price * (1 + stop_pct)
                        take_profit = current_price * (1 - target_pct)
                    
                    risk_per_share = abs(current_price - stop_loss) * self.options_multiplier
                    risk_amount = capital * base_risk
                    shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                    max_shares = capital / (current_price * self.options_multiplier * 2)
                    shares = min(shares, max_shares)
                    
                    if shares > 0.1:
                        positions[symbol] = {
                            'entry_date': current_date,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'shares': shares,
                            'direction': direction,
                            'regime': regime,
                            'signal_type': opp['signal_type'],
                            'quality': opp['quality']
                        }
            
            # Track equity
            unrealized_pnl = sum(
                ((current_prices[sym]['close'] - pos['entry_price']) * pos['shares'] * 
                 self.options_multiplier * (-1 if pos['direction'] == 'SHORT' else 1))
                for sym, pos in positions.items() if sym in current_prices
            )
            equity_curve.append(capital + unrealized_pnl)
        
        # Add to dataframe
        if all_data:
            first_sym = list(all_data.keys())[0]
            all_data[first_sym]['portfolio_equity'] = pd.Series(equity_curve, index=all_dates)
            regime_df = pd.DataFrame(regime_history).set_index('date')
            all_data[first_sym] = all_data[first_sym].join(regime_df, how='left')
        
        return trades, all_data
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Analyze and display results"""
        print("\n" + "="*80)
        print("RESULTS - V5.2")
        print("="*80)
        
        if not trades:
            print("\nNo trades executed.")
            return
        
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['year'] = trades_df['entry_date'].dt.year
        
        total_pnl = trades_df['pnl'].sum()
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winners) / len(trades_df) * 100
        
        # Risk metrics
        first_sym = list(all_data.keys())[0]
        if 'portfolio_equity' in all_data[first_sym].columns:
            equity = all_data[first_sym]['portfolio_equity'].dropna()
            returns = equity.pct_change().dropna()
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            max_dd = drawdown.min()
            years = (equity.index[-1] - equity.index[0]).days / 365.25
            final_capital = self.initial_capital + total_pnl
            cagr = ((final_capital / self.initial_capital) ** (1/years) - 1) * 100
        else:
            sharpe, max_dd, cagr, years = 0, 0, 0, 0
        
        print("\nOVERALL PERFORMANCE:")
        print(f"  Period: {years:.2f} years")
        print(f"  Initial: ${self.initial_capital:,.0f}")
        print(f"  Final: ${self.initial_capital + total_pnl:,.0f}")
        print(f"  Total P&L: ${total_pnl:,.0f}")
        print(f"  Total Return: {(total_pnl/self.initial_capital*100):.1f}%")
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Max DD: {max_dd:.1f}%")
        
        print("\nTRADE STATS:")
        print(f"  Total: {len(trades_df)}")
        print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
        print(f"  Avg Win: ${winners['pnl'].mean():,.0f}" if len(winners) > 0 else "  Avg Win: N/A")
        print(f"  Avg Loss: ${losers['pnl'].mean():,.0f}" if len(losers) > 0 else "  Avg Loss: N/A")
        
        # By regime
        print("\n" + "="*80)
        print("BY REGIME:")
        print("="*80)
        for regime in ['BULL', 'BEAR']:
            reg_trades = trades_df[trades_df['regime'] == regime]
            if len(reg_trades) > 0:
                reg_pnl = reg_trades['pnl'].sum()
                reg_wr = len(reg_trades[reg_trades['pnl'] > 0]) / len(reg_trades) * 100
                print(f"{regime}: {len(reg_trades)} trades, {reg_wr:.1f}% win, ${reg_pnl:,.0f}")
        
        # Yearly
        print("\n" + "="*80)
        print("YEARLY:")
        print("="*80)
        print(f"{'Year':<8} {'Trades':<8} {'Win%':<8} {'P&L':<15} {'Return%':<10}")
        print("-"*60)
        
        cum_cap = self.initial_capital
        for year in sorted(trades_df['year'].unique()):
            yr_trades = trades_df[trades_df['year'] == year]
            yr_pnl = yr_trades['pnl'].sum()
            yr_ret = (yr_pnl / cum_cap) * 100
            yr_wr = len(yr_trades[yr_trades['pnl'] > 0]) / len(yr_trades) * 100
            
            marker = " ← V5 PROBLEM" if year in [2021, 2022] else ""
            print(f"{year:<8} {len(yr_trades):<8} {yr_wr:<8.1f} ${yr_pnl:<14,.0f} {yr_ret:<10.1f}{marker}")
            cum_cap += yr_pnl
        
        # Exit reasons
        print("\n" + "="*80)
        print("EXIT REASONS:")
        print("="*80)
        for reason in trades_df['exit_reason'].value_counts().index:
            reason_trades = trades_df[trades_df['exit_reason'] == reason]
            reason_wr = len(reason_trades[reason_trades['pnl'] > 0]) / len(reason_trades) * 100
            print(f"{reason}: {len(reason_trades)} trades ({reason_wr:.1f}% win)")
        
        # Compare to V5
        stop_trades = trades_df[trades_df['exit_reason'] == 'Stop Loss']
        stop_wr = len(stop_trades[stop_trades['pnl'] > 0]) / len(stop_trades) * 100 if len(stop_trades) > 0 else 0
        
        print("\n" + "="*80)
        print("V5 → V5.2 COMPARISON:")
        print("="*80)
        print(f"{'Metric':<30} {'V5':<15} {'V5.2':<15}")
        print("-"*60)
        print(f"{'Win Rate':<30} {'35.9%':<15} {f'{win_rate:.1f}%':<15}")
        print(f"{'Stop Losses':<30} {'303 (0% win)':<15} {f'{len(stop_trades)} ({stop_wr:.1f}%)':<15}")
        print(f"{'Total Trades':<30} {'493':<15} {len(trades_df)}")
        
        print("\n✅ V5.2 Key Fixes Applied:")
        print("  • Wider stops (2.5% vs 1.5%)")
        print("  • VXX only (removed SQQQ, PG, KO, XLU)")
        print("  • Quality filter (min quality 2)")
        print("  • Better position sizing")
        print("="*80)

def run_v52():
    """Run V5.2 standalone"""
    strategy = RegimeAdaptiveV52(
        bull_symbols=["NVDA", "TSLA", "PLTR", "AMD", "COIN"],
        bear_symbols=["VXX"],
        start_date="2020-01-01",
        initial_capital=10000
    )
    
    all_data = strategy.load_all_data()
    if not all_data:
        print("\n❌ Failed to load data")
        return None, None
    
    all_data = strategy.generate_signals(all_data)
    trades, all_data = strategy.backtest(all_data)
    strategy.analyze_results(trades, all_data)
    
    return trades, all_data

if __name__ == "__main__":
    trades, data = run_v52()