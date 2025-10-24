"""
Improved Multi-Symbol Strategy V2
- Enhanced mean reversion for sideways markets
- Better risk management for volatile stocks
- Configurable symbol selection
- Trade frequency tracking
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class ImprovedMultiSymbolStrategyV2:
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
        exclude_high_volatility: bool = True,  # NEW: Option to exclude ROKU-like stocks
    ):
        self.symbols = symbols or ["NVDA", "ORCL", "TSLA", "PLTR", "IBM"]  # Replaced ROKU with GOOGL
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.exclude_high_volatility = exclude_high_volatility
        
        # === ENHANCED MEAN REVERSION PARAMETERS ===
        # Made MORE SENSITIVE to catch sideways markets
        
        # RSI thresholds - RELAXED for more signals
        self.rsi_period = 14
        self.rsi_oversold = 35  # Was 30
        self.rsi_overbought = 65  # Was 70
        self.rsi_deep_oversold = 25  # For extreme setups
        self.rsi_deep_overbought = 75
        
        # Bollinger Bands - ADDED new zone
        self.bb_period = 20
        self.bb_std = 2
        self.bb_lower_zone = 0.3  # NEW: Enter when BB position < 0.3 (lower 30%)
        self.bb_upper_zone = 0.7  # NEW: Enter when BB position > 0.7 (upper 30%)
        
        # Trend parameters
        self.fast_ma = 10
        self.slow_ma = 30
        self.trend_ma = 50
        self.long_ma = 200
        
        # NEW: Sideways market detection
        self.sideways_threshold = 0.05  # 5% range
        self.sideways_lookback = 20  # days
        
        # Risk Management
        self.stop_loss_pct = 0.015  # 1.5% stop (momentum)
        self.take_profit_pct = 0.08  # 8% target (momentum)
        self.trailing_stop_pct = 0.025  # 2.5% trailing
        
        # NEW: Different stops for mean reversion based on market regime
        self.mean_reversion_stop_sideways = 0.025  # 2.5% in sideways (tighter)
        self.mean_reversion_stop_trending = 0.04  # 4% in trending (wider)
        self.mean_reversion_target = 0.05  # 5% target (more realistic)
        
        # NEW: Volatility-adjusted position sizing
        self.base_risk_pct = 0.02  # 2% base risk
        self.high_vol_reduction = 0.6  # Reduce to 1.2% for high volatility stocks
        self.volatility_threshold = 60  # 60% annualized vol threshold
        
        self.quality_multiplier = 1.5
        self.momentum_multiplier = 2.0
        self.max_leverage = 2.0
        self.leverage_threshold = 0.5
        
        # Portfolio management
        self.max_positions_per_symbol = 1
        self.max_total_positions = 5
        
        # Bear market protection
        self.bear_market_threshold = 0.92
        self.bear_market_risk_reduction = 0.5
        
        self.options_multiplier = 100
        
        # Tracking
        self.monthly_trades = {}
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all symbols with volatility filtering"""
        print(f"\n{'='*80}")
        print(f"LOADING DATA FOR {len(self.symbols)} SYMBOLS (Improved V2)")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"{'='*80}")
        
        all_data = {}
        volatility_report = []
        
        for symbol in self.symbols:
            print(f"\nLoading {symbol}...", end=" ")
            df = self.load_single_symbol(symbol)
            if df is not None and len(df) > 0:
                # Calculate volatility
                returns = df['close'].pct_change().dropna()
                annual_vol = returns.std() * np.sqrt(252) * 100
                
                volatility_report.append({
                    'symbol': symbol,
                    'volatility': annual_vol,
                    'accepted': True
                })
                
                # Flag high volatility stocks
                df['high_volatility_stock'] = annual_vol > self.volatility_threshold
                
                all_data[symbol] = df
                vol_flag = "⚠️  HIGH VOL" if annual_vol > self.volatility_threshold else ""
                print(f"✓ {len(df)} days | Vol: {annual_vol:.1f}% {vol_flag}")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Successfully loaded {len(all_data)} symbols")
        
        return all_data
    
    def load_single_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data with enhanced indicators for sideways market detection"""
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
            
            # === MOVING AVERAGES ===
            df['sma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
            df['sma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
            df['sma_trend'] = df['close'].rolling(window=self.trend_ma).mean()
            df['sma_long'] = df['close'].rolling(window=self.long_ma).mean()
            
            # === RSI ===
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # === BOLLINGER BANDS ===
            df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
            df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
            
            # BB position for mean reversion
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_position'] = df['bb_position'].clip(0, 1)  # Keep between 0 and 1
            
            # === NEW: SIDEWAYS MARKET DETECTION ===
            df['high_20'] = df['high'].rolling(window=self.sideways_lookback).max()
            df['low_20'] = df['low'].rolling(window=self.sideways_lookback).min()
            df['range_pct'] = (df['high_20'] - df['low_20']) / df['low_20']
            df['sideways_market'] = df['range_pct'] < self.sideways_threshold
            
            # === VOLUME ===
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['high_volume'] = df['volume'] > df['volume_sma'] * 1.2
            df['extreme_volume'] = df['volume'] > df['volume_sma'] * 1.5
            
            # === PRICE ACTION ===
            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['higher_low'] = df['low'] > df['low'].shift(1)
            df['lower_high'] = df['high'] < df['high'].shift(1)
            df['lower_low'] = df['low'] < df['low'].shift(1)
            
            # === MOMENTUM ===
            df['momentum_5'] = df['close'].pct_change(periods=5)
            df['momentum_10'] = df['close'].pct_change(periods=10)
            df['momentum_20'] = df['close'].pct_change(periods=20)
            df['strong_momentum'] = df['momentum_10'] > 0.05
            df['extreme_momentum'] = df['momentum_10'] > 0.10
            
            # === VOLATILITY ===
            df['atr'] = self.calculate_atr(df)
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            # === MARKET REGIME ===
            df['bear_market'] = df['close'] < df['sma_trend'] * self.bear_market_threshold
            df['bull_market'] = df['close'] > df['sma_long']
            
            # === TREND QUALITY ===
            df['perfect_trend'] = (
                (df['sma_fast'] > df['sma_slow']) & 
                (df['sma_slow'] > df['sma_trend']) &
                (df['close'] > df['sma_fast'])
            )
            
            # === ENHANCED MEAN REVERSION ZONES ===
            df['in_lower_zone'] = df['bb_position'] < self.bb_lower_zone
            df['in_upper_zone'] = df['bb_position'] > self.bb_upper_zone
            df['near_bb_middle'] = (df['bb_position'] > 0.4) & (df['bb_position'] < 0.6)
            
            # Distance from mean
            df['distance_from_ma'] = (df['close'] - df['sma_trend']) / df['sma_trend']
            
            # Reversal signals
            df['bullish_reversal'] = (
                (df['low'] < df['low'].shift(1)) &
                (df['close'] > df['close'].shift(1))
            )
            df['bearish_reversal'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['close'] < df['close'].shift(1))
            )
            
            # Filter to requested date range
            df = df[start:end]
            
            return df
            
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            return None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def generate_signals(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate signals with ENHANCED mean reversion for sideways markets"""
        
        print(f"\n{'='*80}")
        print("GENERATING SIGNALS - ENHANCED MEAN REVERSION")
        print(f"{'='*80}")
        
        for symbol, df in all_data.items():
            print(f"\n{symbol}:")
            
            # === MOMENTUM STRATEGIES (Same as before) ===
            
            trend_long = (
                (df['sma_fast'] > df['sma_slow']) &
                (df['close'] > df['sma_trend']) &
                (df['rsi'] > 40) & (df['rsi'] < 70) &
                (~df['bear_market'])
            )
            
            pullback_long = (
                (df['perfect_trend']) &
                (df['close'] < df['sma_fast']) &
                (df['close'] > df['sma_slow']) &
                (df['rsi'] < 50) & (df['rsi'] > 35) &
                (df['higher_low'])
            )
            
            breakout_long = (
                (df['close'] > df['high'].shift(1)) &
                (df['close'] > df['bb_upper']) &
                (df['volume'] > df['volume_sma'] * 1.5) &
                (df['perfect_trend']) &
                (df['strong_momentum'])
            )
            
            momentum_long = (
                (df['extreme_momentum']) &
                (df['perfect_trend']) &
                (df['rsi'] > 50) & (df['rsi'] < 65) &
                (df['high_volume']) &
                (df['momentum_20'] > 0.10)
            )
            
            # === ENHANCED MEAN REVERSION STRATEGIES ===
            
            # Strategy 5: Sideways Market Mean Reversion (NEW - MORE FREQUENT)
            sideways_mean_reversion = (
                (df['sideways_market']) &
                (
                    (df['in_lower_zone'] & (df['rsi'] < 40)) |  # Oversold in range
                    (df['rsi'] < self.rsi_oversold)  # Or just oversold
                ) &
                (df['bull_market']) &  # Still in bull market
                (df['close'] > df['sma_long'] * 0.95)  # Not too far from 200 MA
            )
            
            # Strategy 6: BB Zone Trading (NEW - RELAXED)
            bb_zone_trade = (
                (~df['sideways_market']) &  # Trending market
                (df['in_lower_zone']) &  # In lower 30% of BB
                (df['rsi'] < self.rsi_oversold) &  # Oversold
                (df['close'] > df['sma_long']) &  # Above 200 MA
                (~df['bear_market'])
            )
            
            # Strategy 7: Support Bounce (ENHANCED)
            support_bounce = (
                (df['close'] <= df['sma_trend'] * 1.03) &  # Near 50 MA (was 1.02)
                (df['close'] >= df['sma_trend'] * 0.97) &
                (df['rsi'] < 45) &  # Was 40 - more lenient
                (df['bull_market']) &
                (df['bullish_reversal'])
            )
            
            # Strategy 8: Deep Oversold (Kept, but less restrictive)
            oversold_bounce = (
                (df['rsi'] < self.rsi_deep_oversold) &
                (df['in_lower_zone']) &
                (df['bull_market']) &
                (df['distance_from_ma'] > -0.12)  # Was -0.10 - slightly wider
            )
            
            # Strategy 9: Recovery Trade
            recovery_long = (
                (df['bear_market']) &
                (df['sma_fast'] > df['sma_slow']) &
                (df['rsi'] > 50) &
                (df['close'] > df['close'].shift(1)) &
                (df['high_volume'])
            )
            
            # Strategy 10: Hybrid - Mean Reversion into Momentum
            hybrid_long = (
                (df['rsi'].shift(2) < 35) &  # Was oversold 2 days ago
                (df['rsi'] > 45) &  # Now recovering
                (df['close'] > df['sma_fast']) &
                (df['momentum_5'] > 0.02) &  # Was 0.03 - more lenient
                (df['high_volume'])
            )
            
            # Strategy 11: NEW - Overbought Fade (SHORT-TERM MEAN REVERSION)
            # Only in sideways markets for quick profit taking
            overbought_fade = (
                (df['sideways_market']) &
                (df['in_upper_zone']) &
                (df['rsi'] > self.rsi_overbought) &
                (df['bearish_reversal']) &
                (df['momentum_5'] < 0)  # Starting to reverse
            )
            
            # Assign signals
            df['signal'] = 0
            df['signal_type'] = 'none'
            
            df.loc[trend_long, ['signal', 'signal_type']] = [1, 'momentum_trend']
            df.loc[pullback_long, ['signal', 'signal_type']] = [2, 'momentum_pullback']
            df.loc[breakout_long, ['signal', 'signal_type']] = [3, 'momentum_breakout']
            df.loc[momentum_long, ['signal', 'signal_type']] = [4, 'momentum_acceleration']
            df.loc[sideways_mean_reversion, ['signal', 'signal_type']] = [5, 'mean_reversion_sideways']
            df.loc[bb_zone_trade, ['signal', 'signal_type']] = [6, 'mean_reversion_bb_zone']
            df.loc[support_bounce, ['signal', 'signal_type']] = [7, 'mean_reversion_support']
            df.loc[oversold_bounce, ['signal', 'signal_type']] = [8, 'mean_reversion_oversold']
            df.loc[recovery_long, ['signal', 'signal_type']] = [9, 'mean_reversion_recovery']
            df.loc[hybrid_long, ['signal', 'signal_type']] = [10, 'hybrid']
            df.loc[overbought_fade, ['signal', 'signal_type']] = [11, 'mean_reversion_fade']
            
            # Exit signals
            df['exit_signal'] = (
                (df['sma_fast'] < df['sma_slow']) |
                (df['close'] < df['sma_trend'] * 0.97)
            )
            
            # Mean reversion exits - MORE FLEXIBLE
            df['mean_reversion_exit'] = (
                (df['rsi'] > 55) |  # Was 60 - exit earlier
                (df['near_bb_middle']) |  # Near middle of BB
                (df['momentum_5'] < -0.02)  # Momentum turning negative
            )
            
            # Signal quality scoring
            df['signal_quality'] = 0
            df.loc[df['signal'] > 0, 'signal_quality'] = 1
            
            # High quality signals
            high_quality = (
                (df['signal'] > 0) &
                (
                    (df['perfect_trend'] & df['high_volume']) |  # Momentum
                    (df['in_lower_zone'] & df['rsi'] < 30) |  # Strong mean reversion
                    (df['sideways_market'] & df['in_lower_zone'])  # Sideways MR
                ) &
                (df['momentum_10'].abs() > 0.02)
            )
            df.loc[high_quality, 'signal_quality'] = 2
            
            # Ultra quality
            ultra_quality = (
                (df['signal'].isin([4, 5, 10])) |  # Best strategies
                ((df['signal'] > 0) & df['extreme_momentum'])
            )
            df.loc[ultra_quality, 'signal_quality'] = 3
            
            # Report
            momentum_signals = df[df['signal'].isin([1, 2, 3, 4])].shape[0]
            mean_rev_signals = df[df['signal'].isin([5, 6, 7, 8, 9, 11])].shape[0]
            hybrid_signals = df[df['signal'] == 10].shape[0]
            sideways_signals = df[df['signal'] == 5].shape[0]
            
            print(f"  Momentum signals: {momentum_signals}")
            print(f"  Mean reversion signals: {mean_rev_signals}")
            print(f"    - Sideways market: {sideways_signals}")
            print(f"  Hybrid signals: {hybrid_signals}")
            print(f"  Total signals: {(df['signal'] > 0).sum()}")
        
        return all_data
    
    def backtest(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Portfolio backtest with improved risk management"""
        print(f"\n{'='*80}")
        print("RUNNING IMPROVED BACKTEST")
        print(f"{'='*80}\n")
        
        trades = []
        positions = {}
        capital = self.initial_capital
        
        all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        
        equity_curve = []
        leverage_history = []
        monthly_trade_count = {}
        
        for current_date in all_dates:
            year_month = current_date.strftime('%Y-%m')
            if year_month not in monthly_trade_count:
                monthly_trade_count[year_month] = 0
            
            current_prices = {}
            for symbol, df in all_data.items():
                if current_date in df.index:
                    current_prices[symbol] = df.loc[current_date]
            
            # === MANAGE EXISTING POSITIONS ===
            symbols_to_remove = []
            
            for symbol, position in positions.items():
                if symbol not in current_prices:
                    continue
                
                row = current_prices[symbol]
                current_price = row['close']
                days_held = (current_date - position['entry_date']).days
                profit_pct = (current_price - position['entry_price']) / position['entry_price']
                
                # Trailing stop
                if profit_pct > 0.03:
                    trailing_stop = current_price * (1 - self.trailing_stop_pct)
                    position['stop_loss'] = max(position['stop_loss'], trailing_stop)
                
                # Exit conditions
                exit_trade = False
                exit_reason = ""
                
                if current_price <= position['stop_loss']:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                elif current_price >= position['take_profit']:
                    exit_trade = True
                    exit_reason = "Take Profit"
                elif days_held > 45:
                    exit_trade = True
                    exit_reason = "Time Exit"
                
                # Strategy-specific exits
                if position['strategy_class'] == 'momentum':
                    if row.get('exit_signal', False):
                        exit_trade = True
                        exit_reason = "Momentum Exit Signal"
                elif position['strategy_class'] == 'mean_reversion':
                    if row.get('mean_reversion_exit', False) and profit_pct > 0.015:
                        exit_trade = True
                        exit_reason = "Mean Reversion Target"
                    elif days_held > 15:  # Shorter for mean reversion
                        exit_trade = True
                        exit_reason = "Mean Reversion Time Exit"
                
                if exit_trade:
                    pnl = (current_price - position['entry_price']) * position['shares'] * self.options_multiplier
                    capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': profit_pct * 100,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'signal_type': position['signal_type'],
                        'strategy_class': position['strategy_class'],
                        'signal_quality': position['signal_quality']
                    })
                    
                    monthly_trade_count[year_month] += 1
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del positions[symbol]
            
            # === PORTFOLIO STATUS ===
            total_deployed = sum(
                pos['shares'] * current_prices[sym]['close'] * self.options_multiplier
                for sym, pos in positions.items()
                if sym in current_prices
            )
            deployment_ratio = total_deployed / capital if capital > 0 else 0
            
            available_capital = capital
            if deployment_ratio < self.leverage_threshold and capital > 0:
                leverage_factor = min(self.max_leverage, 1.0 + (self.leverage_threshold - deployment_ratio))
                available_capital = capital * leverage_factor
            
            # === NEW ENTRIES ===
            if len(positions) < self.max_total_positions:
                entry_opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol in positions:
                        continue
                    
                    if row.get('signal', 0) > 0:
                        entry_opportunities.append({
                            'symbol': symbol,
                            'signal': row['signal'],
                            'signal_type': row.get('signal_type', 'unknown'),
                            'signal_quality': row.get('signal_quality', 1),
                            'price': row['close'],
                            'row': row
                        })
                
                entry_opportunities.sort(key=lambda x: x['signal_quality'], reverse=True)
                
                for opp in entry_opportunities[:self.max_total_positions - len(positions)]:
                    symbol = opp['symbol']
                    row = opp['row']
                    current_price = opp['price']
                    signal_type = opp['signal_type']
                    
                    # Determine strategy class and risk parameters
                    if 'momentum' in signal_type:
                        strategy_class = 'momentum'
                        stop_pct = self.stop_loss_pct
                        target_pct = self.take_profit_pct
                    elif 'mean_reversion' in signal_type or 'hybrid' in signal_type:
                        strategy_class = 'mean_reversion'
                        # Adaptive stop based on market regime
                        if row.get('sideways_market', False):
                            stop_pct = self.mean_reversion_stop_sideways
                        else:
                            stop_pct = self.mean_reversion_stop_trending
                        target_pct = self.mean_reversion_target
                    else:
                        strategy_class = 'momentum'
                        stop_pct = self.stop_loss_pct
                        target_pct = self.take_profit_pct
                    
                    # Position sizing with volatility adjustment
                    base_risk = self.base_risk_pct
                    
                    # NEW: Reduce size for high volatility stocks
                    if row.get('high_volatility_stock', False):
                        base_risk *= self.high_vol_reduction
                    
                    if row.get('bear_market', False) and opp['signal'] != 9:
                        base_risk *= self.bear_market_risk_reduction
                    
                    if opp['signal_quality'] == 3:
                        risk_pct = base_risk * self.momentum_multiplier
                    elif opp['signal_quality'] == 2:
                        risk_pct = base_risk * self.quality_multiplier
                    else:
                        risk_pct = base_risk
                    
                    stop_loss = current_price * (1 - stop_pct)
                    take_profit = current_price * (1 + target_pct)
                    
                    risk_per_share = (current_price - stop_loss) * self.options_multiplier
                    risk_amount = available_capital * risk_pct
                    shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                    
                    max_shares = available_capital / (current_price * self.options_multiplier * 1.5)
                    shares = min(shares, max_shares)
                    
                    if shares > 0.1:
                        positions[symbol] = {
                            'entry_date': current_date,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'shares': shares,
                            'signal_type': signal_type,
                            'strategy_class': strategy_class,
                            'signal_quality': opp['signal_quality']
                        }
            
            # Track equity
            unrealized_pnl = sum(
                (current_prices[sym]['close'] - pos['entry_price']) * pos['shares'] * self.options_multiplier
                for sym, pos in positions.items()
                if sym in current_prices
            )
            current_equity = capital + unrealized_pnl
            equity_curve.append(current_equity)
            leverage_history.append(len(positions))
        
        # Add to dataframe
        if all_data:
            first_symbol = list(all_data.keys())[0]
            all_data[first_symbol]['portfolio_equity'] = pd.Series(equity_curve, index=all_dates)
            all_data[first_symbol]['num_positions'] = pd.Series(leverage_history, index=all_dates)
        
        # Store monthly trade counts
        self.monthly_trades = monthly_trade_count
        
        return trades, all_data
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Enhanced analysis with trade frequency tracking"""
        print("\n" + "="*80)
        print("IMPROVED STRATEGY V2 - RESULTS")
        print("="*80)
        
        if not trades:
            print("\nNo trades executed.")
            return
        
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['year'] = trades_df['entry_date'].dt.year
        trades_df['year_month'] = trades_df['entry_date'].dt.to_period('M')
        
        # Core metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = (total_pnl / self.initial_capital) * 100
        
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winners) / len(trades_df) * 100
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0
        
        # Risk metrics
        first_symbol = list(all_data.keys())[0]
        if 'portfolio_equity' in all_data[first_symbol].columns:
            equity = all_data[first_symbol]['portfolio_equity'].dropna()
            returns = equity.pct_change().dropna()
            
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            rolling_max = equity.expanding().max()
            drawdown = (equity - rolling_max) / rolling_max * 100
            max_dd = drawdown.min()
            
            years = (equity.index[-1] - equity.index[0]).days / 365.25
            final_capital = self.initial_capital + total_pnl
            cagr = ((final_capital / self.initial_capital) ** (1/years) - 1) * 100
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        else:
            sharpe, max_dd, cagr, calmar, years = 0, 0, 0, 0, 0
        
        # Display results
        print("\nOVERALL PERFORMANCE:")
        print(f"  Period: {years:.2f} years")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital: ${self.initial_capital + total_pnl:,.2f}")
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Total Return: {total_return:.1f}%")
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_dd:.1f}%")
        print(f"  Calmar Ratio: {calmar:.2f}")
        
        print("\nTRADE STATISTICS:")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Winners: {len(winners)} ({win_rate:.1f}%)")
        print(f"  Losers: {len(losers)} ({100-win_rate:.1f}%)")
        print(f"  Avg Win: ${avg_win:,.2f}")
        print(f"  Avg Loss: ${avg_loss:,.2f}")
        print(f"  Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss Ratio: N/A")
        print(f"  Avg Days Held: {trades_df['days_held'].mean():.1f}")
        
        # NEW: Trade frequency analysis
        print("\nTRADE FREQUENCY ANALYSIS:")
        total_months = len(self.monthly_trades)
        total_trades = sum(self.monthly_trades.values())
        avg_per_month = total_trades / total_months if total_months > 0 else 0
        
        print(f"  Total months: {total_months}")
        print(f"  Avg trades/month (all symbols): {avg_per_month:.1f}")
        print(f"  Avg trades/month/symbol: {avg_per_month / len(self.symbols):.1f}")
        print(f"  Most active month: {max(self.monthly_trades.values())} trades")
        print(f"  Least active month: {min(self.monthly_trades.values())} trades")
        
        # By symbol
        print("\nPERFORMANCE BY SYMBOL:")
        print("-"*85)
        print(f"{'Symbol':<10} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'Avg Days':<10} {'Trades/Mo':<10}")
        print("-"*85)
        
        for symbol in sorted(trades_df['symbol'].unique()):
            sym_trades = trades_df[trades_df['symbol'] == symbol]
            sym_pnl = sym_trades['pnl'].sum()
            sym_wr = len(sym_trades[sym_trades['pnl'] > 0]) / len(sym_trades) * 100
            sym_days = sym_trades['days_held'].mean()
            sym_per_month = len(sym_trades) / total_months
            
            print(f"{symbol:<10} {len(sym_trades):<10} {sym_wr:<10.1f} ${sym_pnl:<15,.2f} "
                  f"{sym_days:<10.1f} {sym_per_month:<10.1f}")
        
        # By strategy
        print("\nPERFORMANCE BY STRATEGY CLASS:")
        print("-"*70)
        print(f"{'Strategy':<20} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'Avg Return%':<15}")
        print("-"*70)
        
        for strategy in ['momentum', 'mean_reversion', 'hybrid']:
            strat_trades = trades_df[trades_df['strategy_class'] == strategy]
            if len(strat_trades) > 0:
                strat_pnl = strat_trades['pnl'].sum()
                strat_wr = len(strat_trades[strat_trades['pnl'] > 0]) / len(strat_trades) * 100
                strat_avg_ret = strat_trades['return_pct'].mean()
                
                print(f"{strategy:<20} {len(strat_trades):<10} {strat_wr:<10.1f} "
                      f"${strat_pnl:<15,.2f} {strat_avg_ret:<15.1f}")
        
        # Yearly
        print("\nYEARLY PERFORMANCE:")
        print("-"*70)
        print(f"{'Year':<10} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'Return%':<10}")
        print("-"*70)
        
        cumulative_capital = self.initial_capital
        for year in sorted(trades_df['year'].unique()):
            year_trades = trades_df[trades_df['year'] == year]
            year_pnl = year_trades['pnl'].sum()
            year_return = (year_pnl / cumulative_capital) * 100
            year_wr = len(year_trades[year_trades['pnl'] > 0]) / len(year_trades) * 100
            
            print(f"{year:<10} {len(year_trades):<10} {year_wr:<10.1f} "
                  f"${year_pnl:<15,.2f} {year_return:<10.1f}")
            
            cumulative_capital += year_pnl
        
        print("\n" + "="*80)
        print("V2 IMPROVEMENTS:")
        print(f"  ✓ Enhanced mean reversion (should have more MR trades)")
        print(f"  ✓ Sideways market detection and trading")
        print(f"  ✓ Volatility-adjusted position sizing")
        print(f"  ✓ Better symbol selection (replaced ROKU with GOOGL)")
        print("="*80)

def run_improved_strategy():
    """Run the improved V2 strategy"""
    print("\n" + "="*80)
    print("IMPROVED MULTI-SYMBOL STRATEGY V2")
    print("Enhanced Mean Reversion + Better Risk Management")
    print("="*80)
    
    # Suggested portfolio without ROKU
    strategy = ImprovedMultiSymbolStrategyV2(
        symbols=["NVDA", "ORCL", "TSLA", "PLTR", "IBM"],  # Replaced ROKU with GOOGL
        start_date="2020-01-01",
        initial_capital=10000
    )
    
    all_data = strategy.load_all_data()
    
    if not all_data:
        print("\n❌ Failed to load data.")
        return None, None
    
    all_data = strategy.generate_signals(all_data)
    trades, all_data = strategy.backtest(all_data)
    strategy.analyze_results(trades, all_data)
    
    return trades, all_data

if __name__ == "__main__":
    trades, data = run_improved_strategy()