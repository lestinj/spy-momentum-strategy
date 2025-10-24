"""
Enhanced Dynamic Allocation Strategy V4.7
═══════════════════════════════════════════════════════════════════════════════

MAJOR IMPROVEMENTS FROM V4.6:
  1. ✓ More aggressive VXX entry (no quality filter required)
  2. ✓ Increased bear positions (4 max instead of 2)
  3. ✓ Stronger allocation ratios (20/80 in BEAR regime)
  4. ✓ Added inverse ETFs (SQQQ, SPXU) for more downside tools
  5. ✓ Faster regime detection (3 days vs 5)
  6. ✓ Emergency protection mode (100% bear on crashes)

DYNAMIC ALLOCATION:
  - BULL regime: 80% bull / 20% bear (hedge)
  - NEUTRAL: 50% bull / 50% bear (balanced)
  - BEAR regime: 20% bull / 80% bear (STRONG protection)
  - EMERGENCY: 0% bull / 100% bear (crash mode)
  
BULL POSITIONS:
  - Symbols: NVDA, TSLA, PLTR, AMD, COIN
  - Max positions: 4
  - Stop loss: 1.5%
  - Position sizing: 2-4%
  
BEAR POSITIONS (ENHANCED):
  - Symbols: VXX, SQQQ, SPXU (3 tools for downside)
  - Max positions: 4 (increased from 2)
  - Stop loss: 2.0%
  - Position sizing: 2-3%
  - NO quality filter (trade all signals)
  
Expected Performance:
  - Bull years: 75-85% CAGR
  - Bear years: +20-30% (MUCH better protection)
  - Max DD: -20% to -30% (target improvement)
  - 2022: Target +15-25% (vs V4.6's -21.8%)
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class EnhancedDynamicStrategy:
    """V4.7 - Enhanced dynamic allocation with aggressive bear protection"""
    
    def __init__(
        self,
        bull_symbols: List[str] = None,
        bear_symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
    ):
        self.bull_symbols = bull_symbols or ["NVDA", "TSLA", "PLTR", "AMD", "COIN"]
        # ENHANCEMENT 4: Added SQQQ and SPXU
        self.bear_symbols = bear_symbols or ["VXX", "SQQQ", "SPXU"]
        
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
        
        # === ENHANCED ALLOCATION PARAMETERS ===
        print("\n" + "="*80)
        print("ENHANCED DYNAMIC ALLOCATION STRATEGY V4.7")
        print("Maximum downside protection with dynamic allocation")
        print("="*80)
        
        # ENHANCEMENT 3: Stronger allocation in BEAR, added EMERGENCY
        self.allocation_schemes = {
            'BULL': {'bull': 0.80, 'bear': 0.20},      # 80% bull, 20% bear hedge
            'NEUTRAL': {'bull': 0.50, 'bear': 0.50},   # 50/50 balanced
            'BEAR': {'bull': 0.20, 'bear': 0.80},      # 20% bull, 80% BEAR (stronger)
            'EMERGENCY': {'bull': 0.00, 'bear': 1.00}  # 100% bear on crashes
        }
        
        print("\n📊 ENHANCED DYNAMIC ALLOCATION:")
        print("  BULL regime:      80% bull / 20% bear (hedge)")
        print("  NEUTRAL regime:   50% bull / 50% bear (balanced)")
        print("  BEAR regime:      20% bull / 80% bear (STRONG protection)")
        print("  EMERGENCY mode:   0% bull / 100% bear (crash protection)")
        
        # === BULL POSITION PARAMETERS ===
        print("\n📈 BULL POSITIONS:")
        self.bull_base_risk = 0.020        # 2% base
        self.bull_quality_mult = 1.5       # 3% on quality
        self.bull_momentum_mult = 2.0      # 4% on momentum
        self.bull_stop_loss = 0.015        # 1.5%
        self.bull_take_profit = 0.08       # 8%
        self.bull_max_hold = 45
        self.bull_max_positions = 4
        
        print(f"  Symbols: {', '.join(self.bull_symbols)}")
        print(f"  Max positions: {self.bull_max_positions}")
        print(f"  Stop loss: {self.bull_stop_loss*100}%")
        
        # === ENHANCED BEAR POSITION PARAMETERS ===
        print("\n🛡️ BEAR POSITIONS (ENHANCED):")
        self.bear_base_risk = 0.020        # 2% base
        self.bear_quality_mult = 1.3       # 2.6% on quality
        self.bear_stop_loss = 0.020        # 2%
        self.bear_take_profit = 0.15       # 15%
        self.bear_max_hold = 60
        # ENHANCEMENT 2: Increased from 2 to 4
        self.bear_max_positions = 4
        self.bear_fast_exit_days = 20
        # ENHANCEMENT 1: More aggressive entry (will remove quality filter in code)
        self.bear_require_quality = False  # Trade ALL bear signals
        
        print(f"  Symbols: {', '.join(self.bear_symbols)}")
        print(f"  Max positions: {self.bear_max_positions} (increased from 2)")
        print(f"  Quality filter: REMOVED (trade all signals)")
        print(f"  Stop loss: {self.bear_stop_loss*100}%")
        
        # === ENHANCED REGIME DETECTION ===
        print("\n🔄 REGIME DETECTION (ENHANCED):")
        # ENHANCEMENT 5: Faster detection (3 days vs 5)
        self.regime_confirmation_days = 3
        self.current_regime = "NEUTRAL"
        self.regime_switch_pending = False
        self.regime_switch_date = None
        
        # Emergency protection trigger
        self.emergency_drawdown_threshold = -0.10  # -10% from recent high
        self.emergency_active = False
        
        print(f"  Confirmation period: {self.regime_confirmation_days} days (faster)")
        print(f"  Emergency trigger: -10% from recent high")
        print(f"  4 states: BULL, NEUTRAL, BEAR, EMERGENCY")
        
        self.options_multiplier = 100
        self.spy_data = None
        self.spy_recent_high = None
        
        print("\n" + "="*80)
        print("V4.7 KEY IMPROVEMENTS:")
        print("  ✓ More bear tools (VXX, SQQQ, SPXU)")
        print("  ✓ 4 bear positions (vs 2 in V4.6)")
        print("  ✓ 80% bear allocation in BEAR regime")
        print("  ✓ NO quality filter on bear signals")
        print("  ✓ Faster regime detection (3 days)")
        print("  ✓ Emergency crash protection mode")
        print("="*80)
        print("\nEXPECTED IMPROVEMENTS:")
        print("  • Max DD: -20% to -30% (vs V4.6's -57.8%)")
        print("  • 2022: +15-25% (vs V4.6's -21.8%)")
        print("  • More consistent bear protection")
        print("="*80 + "\n")
    
    def load_spy_data(self) -> pd.DataFrame:
        """Load SPY for enhanced regime detection"""
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
            
            # Enhanced regime detection
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            
            # Multiple regime signals
            df['below_200ma'] = df['close'] < df['sma_200']
            df['distance_from_200'] = (df['close'] - df['sma_200']) / df['sma_200']
            df['death_cross'] = df['sma_50'] < df['sma_200']
            df['momentum_negative'] = df['close'].pct_change(20) < -0.05
            
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252) * 100
            df['high_volatility'] = df['volatility_20'] > 25
            
            # Recent high for emergency detection
            df['recent_high'] = df['close'].rolling(60).max()
            df['drawdown_from_high'] = (df['close'] - df['recent_high']) / df['recent_high']
            df['is_crash'] = df['drawdown_from_high'] < self.emergency_drawdown_threshold
            
            # Score-based regime (0-4 bear signals)
            df['bear_signals'] = (
                df['below_200ma'].astype(int) +
                df['death_cross'].astype(int) +
                df['momentum_negative'].astype(int) +
                df['high_volatility'].astype(int)
            )
            
            # 4-state regime (added EMERGENCY)
            df['regime'] = 'NEUTRAL'
            df.loc[df['bear_signals'] >= 3, 'regime'] = 'BEAR'
            df.loc[df['bear_signals'] <= 1, 'regime'] = 'BULL'
            df.loc[df['is_crash'], 'regime'] = 'EMERGENCY'  # Override with emergency
            
            df = df[start:end]
            
            regime_counts = df['regime'].value_counts()
            print(f"  ✓ SPY loaded: {len(df)} days")
            for regime in ['BULL', 'NEUTRAL', 'BEAR', 'EMERGENCY']:
                if regime in regime_counts:
                    pct = regime_counts[regime] / len(df) * 100
                    print(f"    {regime}: {regime_counts[regime]} days ({pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def detect_regime(self, date: datetime) -> str:
        """Detect regime with faster confirmation and emergency mode"""
        if self.spy_data is None or date not in self.spy_data.index:
            return self.current_regime
        
        spy_row = self.spy_data.loc[date]
        new_regime = spy_row.get('regime', 'NEUTRAL')
        
        # Emergency mode overrides everything immediately (no confirmation needed)
        if new_regime == 'EMERGENCY':
            if not self.emergency_active:
                print(f"\n🚨 EMERGENCY MODE ACTIVATED on {date.date()}")
                print("   SPY dropped >10% from recent high - Going 100% BEAR")
                self.emergency_active = True
            self.current_regime = 'EMERGENCY'
            return 'EMERGENCY'
        else:
            if self.emergency_active:
                print(f"\n✅ EMERGENCY MODE DEACTIVATED on {date.date()}")
                self.emergency_active = False
        
        # Normal regime detection with confirmation
        if new_regime != self.current_regime:
            if not self.regime_switch_pending:
                self.regime_switch_pending = True
                self.regime_switch_date = date
                return self.current_regime
            else:
                days_pending = (date - self.regime_switch_date).days
                if days_pending >= self.regime_confirmation_days:
                    print(f"\n🔄 REGIME CHANGE: {self.current_regime} → {new_regime} on {date.date()}")
                    old_alloc = self.allocation_schemes[self.current_regime]
                    new_alloc = self.allocation_schemes[new_regime]
                    print(f"   Allocation: {old_alloc['bull']:.0%}/{old_alloc['bear']:.0%} → {new_alloc['bull']:.0%}/{new_alloc['bear']:.0%} (bull/bear)")
                    
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
            df['extreme_volume'] = df['volume'] > df['volume_sma'] * 1.5
            
            # Price action
            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['higher_low'] = df['low'] > df['low'].shift(1)
            
            # Momentum
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            df['strong_momentum'] = df['momentum_10'] > 0.05
            df['extreme_momentum'] = df['momentum_10'] > 0.10
            
            # Trends
            df['uptrend'] = (df['sma_fast'] > df['sma_slow']) & (df['sma_slow'] > df['sma_trend'])
            df['downtrend'] = (df['sma_fast'] < df['sma_slow']) & (df['sma_slow'] < df['sma_trend'])
            df['perfect_uptrend'] = df['uptrend'] & (df['close'] > df['sma_fast'])
            df['perfect_downtrend'] = df['downtrend'] & (df['close'] < df['sma_fast'])
            
            # Market regime
            df['bear_market'] = df['close'] < df['sma_trend'] * 0.92
            df['bull_market'] = df['close'] > df['sma_long']
            
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
        
        self.spy_data = self.load_spy_data()
        
        all_data = {}
        all_symbols = list(set(self.bull_symbols + self.bear_symbols))
        
        for symbol in all_symbols:
            print(f"Loading {symbol}...", end=" ")
            df = self.load_single_symbol(symbol)
            if df is not None and len(df) > 0:
                if self.spy_data is not None:
                    df = df.join(
                        self.spy_data[['regime']], 
                        how='left'
                    )
                    df['spy_regime'] = df['regime'].fillna('NEUTRAL')
                else:
                    df['spy_regime'] = 'NEUTRAL'
                
                all_data[symbol] = df
                print(f"✓ {len(df)} days")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Successfully loaded {len(all_data)} symbols\n")
        return all_data
    
    def generate_bull_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate bull momentum signals"""
        
        # Strategy 1: Classic Trend
        trend_long = (
            (df['sma_fast'] > df['sma_slow']) &
            (df['close'] > df['sma_trend']) &
            (df['rsi'] > 40) & (df['rsi'] < 70) &
            (~df['bear_market'])
        )
        
        # Strategy 2: Quality Pullback
        pullback_long = (
            df['perfect_uptrend'] &
            (df['close'] < df['sma_fast']) &
            (df['close'] > df['sma_slow']) &
            (df['rsi'] < 50) & (df['rsi'] > 35) &
            df['higher_low']
        )
        
        # Strategy 3: Breakout
        breakout_long = (
            (df['close'] > df['high'].shift(1)) &
            (df['close'] > df['bb_upper']) &
            (df['volume'] > df['volume_sma'] * 1.5) &
            df['perfect_uptrend'] &
            df['strong_momentum']
        )
        
        # Strategy 4: Momentum Acceleration
        momentum_long = (
            df['extreme_momentum'] &
            df['perfect_uptrend'] &
            (df['rsi'] > 50) & (df['rsi'] < 65) &
            df['high_volume'] &
            (df['momentum_20'] > 0.10)
        )
        
        # Strategy 5: Oversold Bounce
        bounce_long = (
            (df['close'] > df['sma_trend'] * 0.97) &
            (df['rsi'] < 30) &
            (df['close'] < df['bb_lower']) &
            (~df['bear_market'])
        )
        
        df['bull_signal'] = 0
        df['bull_signal_type'] = 'none'
        
        df.loc[trend_long, ['bull_signal', 'bull_signal_type']] = [1, 'trend']
        df.loc[pullback_long, ['bull_signal', 'bull_signal_type']] = [2, 'pullback']
        df.loc[breakout_long, ['bull_signal', 'bull_signal_type']] = [3, 'breakout']
        df.loc[momentum_long, ['bull_signal', 'bull_signal_type']] = [4, 'momentum']
        df.loc[bounce_long, ['bull_signal', 'bull_signal_type']] = [5, 'bounce']
        
        # Exit signal
        df['bull_exit_signal'] = (
            (df['sma_fast'] < df['sma_slow']) |
            (df['close'] < df['sma_trend'] * 0.97)
        )
        
        # Quality scoring
        df['bull_quality'] = 1
        df.loc[df['bull_signal'] > 0, 'bull_quality'] = 1
        
        high_quality = (
            (df['bull_signal'] > 0) &
            df['perfect_uptrend'] &
            df['high_volume'] &
            (df['momentum_10'] > 0.03)
        )
        df.loc[high_quality, 'bull_quality'] = 2
        
        ultra_quality = (
            (df['bull_signal'].isin([4])) |
            ((df['bull_signal'] > 0) & df['extreme_momentum'])
        )
        df.loc[ultra_quality, 'bull_quality'] = 3
        
        return df
    
    def generate_bear_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate bear signals (VXX, SQQQ, SPXU) - AGGRESSIVE"""
        
        # More aggressive bear signals - multiple conditions
        
        # Signal 1: Downtrend (relaxed)
        trend_short = (
            df['downtrend'] &
            (df['rsi'] < 65)  # Relaxed from 60
        )
        
        # Signal 2: Breakdown (relaxed)
        breakdown_short = (
            (df['close'] < df['bb_lower']) &
            df['perfect_downtrend']
        )
        
        # Signal 3: Momentum reversal (new)
        momentum_short = (
            (df['momentum_10'] < -0.03) &
            (df['close'] < df['sma_slow'])
        )
        
        # Signal 4: Volume spike down (new)
        volume_short = (
            df['extreme_volume'] &
            (df['close'] < df['close'].shift(1)) &
            (df['rsi'] < 50)
        )
        
        df['bear_signal'] = 0
        df['bear_signal_type'] = 'none'
        
        df.loc[trend_short, ['bear_signal', 'bear_signal_type']] = [1, 'trend']
        df.loc[breakdown_short, ['bear_signal', 'bear_signal_type']] = [2, 'breakdown']
        df.loc[momentum_short, ['bear_signal', 'bear_signal_type']] = [3, 'momentum']
        df.loc[volume_short, ['bear_signal', 'bear_signal_type']] = [4, 'volume']
        
        # Quality scoring (but won't be used for filtering)
        df['bear_quality'] = 0
        df.loc[df['bear_signal'] > 0, 'bear_quality'] = 1
        
        high_quality_bear = (
            (df['bear_signal'] > 0) &
            df['perfect_downtrend'] &
            df['high_volume'] &
            (df['momentum_10'] < -0.05)
        )
        df.loc[high_quality_bear, 'bear_quality'] = 2
        
        return df
    
    def generate_signals(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate signals for all symbols"""
        print("="*80)
        print("GENERATING SIGNALS")
        print("="*80)
        
        for symbol, df in all_data.items():
            if symbol in self.bull_symbols:
                df = self.generate_bull_signals(df)
                bull_sigs = (df['bull_signal'] > 0).sum()
                print(f"{symbol}: {bull_sigs} bull signals")
            
            if symbol in self.bear_symbols:
                df = self.generate_bear_signals(df)
                bear_sigs = (df['bear_signal'] > 0).sum()
                print(f"{symbol}: {bear_sigs} bear signals (NO quality filter)")
            
            all_data[symbol] = df
        
        print()
        return all_data
    
    def backtest(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run enhanced backtest with aggressive bear protection"""
        print("="*80)
        print("RUNNING ENHANCED BACKTEST")
        print("Aggressive bear protection with 4 positions")
        print("="*80 + "\n")
        
        trades = []
        positions = {}
        capital = self.initial_capital
        
        all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        equity_curve = []
        regime_history = []
        allocation_history = []
        
        for current_date in all_dates:
            regime = self.detect_regime(current_date)
            regime_history.append({'date': current_date, 'current_regime': regime})
            
            # Get current allocation scheme
            bull_alloc = self.allocation_schemes[regime]['bull']
            bear_alloc = self.allocation_schemes[regime]['bear']
            allocation_history.append({
                'date': current_date,
                'bull_allocation': bull_alloc,
                'bear_allocation': bear_alloc
            })
            
            # Get current prices
            current_prices = {}
            for symbol in self.bull_symbols + self.bear_symbols:
                if symbol in all_data and current_date in all_data[symbol].index:
                    current_prices[symbol] = all_data[symbol].loc[current_date]
            
            # === MANAGE EXISTING POSITIONS ===
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
                
                # Get exit parameters
                if pos['direction'] == 'LONG':
                    max_hold = self.bull_max_hold
                    exit_signal = row.get('bull_exit_signal', False)
                else:
                    max_hold = self.bear_max_hold
                    exit_signal = False
                    
                    # VXX fast exit logic
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
                
                # Standard exit logic
                exit_trade = False
                exit_reason = ""
                
                if pos['direction'] == 'LONG':
                    if current_price <= pos['stop_loss']:
                        exit_trade, exit_reason = True, "Stop Loss"
                    elif current_price >= pos['take_profit']:
                        exit_trade, exit_reason = True, "Take Profit"
                    elif exit_signal:
                        exit_trade, exit_reason = True, "Exit Signal"
                    
                    # Emergency: exit all longs immediately
                    if regime == 'EMERGENCY':
                        exit_trade, exit_reason = True, "Emergency Exit"
                        
                else:  # SHORT
                    if current_price >= pos['stop_loss']:
                        exit_trade, exit_reason = True, "Stop Loss"
                    elif current_price <= pos['take_profit']:
                        exit_trade, exit_reason = True, "Take Profit"
                
                if days_held > max_hold:
                    exit_trade, exit_reason = True, "Time Exit"
                
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
            
            # === COUNT CURRENT POSITIONS BY TYPE ===
            bull_positions = sum(1 for p in positions.values() if p['direction'] == 'LONG')
            bear_positions = sum(1 for p in positions.values() if p['direction'] == 'SHORT')
            
            # === NEW BULL ENTRIES (unless EMERGENCY) ===
            if regime != 'EMERGENCY' and bull_positions < self.bull_max_positions:
                bull_opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol not in self.bull_symbols or symbol in positions:
                        continue
                    
                    if row.get('bull_signal', 0) > 0:
                        bull_opportunities.append({
                            'symbol': symbol,
                            'quality': row.get('bull_quality', 1),
                            'price': row['close'],
                            'direction': 'LONG',
                            'signal_type': row.get('bull_signal_type', 'unknown')
                        })
                
                bull_opportunities.sort(key=lambda x: x['quality'], reverse=True)
                
                for opp in bull_opportunities[:self.bull_max_positions - bull_positions]:
                    symbol = opp['symbol']
                    current_price = opp['price']
                    
                    # Position sizing with allocation adjustment
                    base_risk = self.bull_base_risk
                    if opp['quality'] == 2:
                        base_risk *= self.bull_quality_mult
                    elif opp['quality'] == 3:
                        base_risk *= self.bull_momentum_mult
                    
                    # Apply regime allocation
                    base_risk *= bull_alloc
                    
                    stop_loss = current_price * (1 - self.bull_stop_loss)
                    take_profit = current_price * (1 + self.bull_take_profit)
                    
                    risk_per_share = (current_price - stop_loss) * self.options_multiplier
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
                            'direction': 'LONG',
                            'regime': regime,
                            'signal_type': opp['signal_type'],
                            'quality': opp['quality']
                        }
            
            # === NEW BEAR ENTRIES (AGGRESSIVE - NO QUALITY FILTER) ===
            if bear_positions < self.bear_max_positions:
                bear_opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol not in self.bear_symbols or symbol in positions:
                        continue
                    
                    # ENHANCEMENT 1: NO quality filter - trade ALL signals
                    if row.get('bear_signal', 0) > 0:
                        bear_opportunities.append({
                            'symbol': symbol,
                            'quality': row.get('bear_quality', 1),
                            'price': row['close'],
                            'direction': 'SHORT',
                            'signal_type': row.get('bear_signal_type', 'unknown')
                        })
                
                bear_opportunities.sort(key=lambda x: x['quality'], reverse=True)
                
                for opp in bear_opportunities[:self.bear_max_positions - bear_positions]:
                    symbol = opp['symbol']
                    current_price = opp['price']
                    
                    # Position sizing with allocation adjustment
                    base_risk = self.bear_base_risk
                    if opp['quality'] == 2:
                        base_risk *= self.bear_quality_mult
                    
                    # Apply regime allocation
                    base_risk *= bear_alloc
                    
                    stop_loss = current_price * (1 + self.bear_stop_loss)
                    take_profit = current_price * (1 - self.bear_take_profit)
                    
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
                            'direction': 'SHORT',
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
            
            # Add regime tracking
            regime_df = pd.DataFrame(regime_history).set_index('date')
            all_data[first_sym] = all_data[first_sym].join(regime_df, how='left')
            
            # Add allocation tracking
            alloc_df = pd.DataFrame(allocation_history).set_index('date')
            all_data[first_sym] = all_data[first_sym].join(alloc_df, how='left')
        
        return trades, all_data
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Analyze results"""
        print("\n" + "="*80)
        print("ENHANCED DYNAMIC STRATEGY V4.7 RESULTS")
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
        print(f"  Total Return: {(total_pnl/self.initial_capital*100):.1f}%")
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Max DD: {max_dd:.1f}%")
        print(f"  Win Rate: {win_rate:.1f}%")
        
        # By direction
        print("\n" + "="*80)
        print("BY POSITION TYPE:")
        print("="*80)
        for direction in ['LONG', 'SHORT']:
            dir_name = "Bull" if direction == 'LONG' else "Bear"
            dir_trades = trades_df[trades_df['direction'] == direction]
            if len(dir_trades) > 0:
                dir_pnl = dir_trades['pnl'].sum()
                dir_wr = len(dir_trades[dir_trades['pnl'] > 0]) / len(dir_trades) * 100
                dir_contrib = (dir_pnl / total_pnl * 100) if total_pnl != 0 else 0
                avg_return = dir_trades['return_pct'].mean()
                print(f"{dir_name}: {len(dir_trades)} trades, {dir_wr:.1f}% win, "
                      f"${dir_pnl:,.0f} ({dir_contrib:.1f}%), avg {avg_return:.1f}%")
        
        # By symbol (bear symbols)
        print("\n" + "="*80)
        print("BEAR SYMBOLS BREAKDOWN:")
        print("="*80)
        bear_trades = trades_df[trades_df['direction'] == 'SHORT']
        for symbol in self.bear_symbols:
            sym_trades = bear_trades[bear_trades['symbol'] == symbol]
            if len(sym_trades) > 0:
                sym_pnl = sym_trades['pnl'].sum()
                sym_wr = len(sym_trades[sym_trades['pnl'] > 0]) / len(sym_trades) * 100
                print(f"{symbol}: {len(sym_trades)} trades, {sym_wr:.1f}% win, ${sym_pnl:,.0f}")
        
        # By regime
        print("\n" + "="*80)
        print("BY REGIME:")
        print("="*80)
        for regime in ['BULL', 'NEUTRAL', 'BEAR', 'EMERGENCY']:
            reg_trades = trades_df[trades_df['regime'] == regime]
            if len(reg_trades) > 0:
                reg_pnl = reg_trades['pnl'].sum()
                reg_wr = len(reg_trades[reg_trades['pnl'] > 0]) / len(reg_trades) * 100
                
                # Break down by direction
                reg_long = reg_trades[reg_trades['direction'] == 'LONG']
                reg_short = reg_trades[reg_trades['direction'] == 'SHORT']
                
                print(f"\n{regime}:")
                print(f"  Total: {len(reg_trades)} trades, {reg_wr:.1f}% win, ${reg_pnl:,.0f}")
                if len(reg_long) > 0:
                    long_pnl = reg_long['pnl'].sum()
                    long_wr = len(reg_long[reg_long['pnl'] > 0]) / len(reg_long) * 100
                    print(f"    Bull: {len(reg_long)} trades, {long_wr:.1f}% win, ${long_pnl:,.0f}")
                if len(reg_short) > 0:
                    short_pnl = reg_short['pnl'].sum()
                    short_wr = len(reg_short[reg_short['pnl'] > 0]) / len(reg_short) * 100
                    print(f"    Bear: {len(reg_short)} trades, {short_wr:.1f}% win, ${short_pnl:,.0f}")
        
        # Yearly
        print("\n" + "="*80)
        print("YEARLY PERFORMANCE:")
        print("="*80)
        print(f"{'Year':<8} {'Trades':<8} {'Win%':<8} {'P&L':<15} {'Return%':<10}")
        print("-"*60)
        
        cum_cap = self.initial_capital
        for year in sorted(trades_df['year'].unique()):
            yr_trades = trades_df[trades_df['year'] == year]
            yr_pnl = yr_trades['pnl'].sum()
            yr_ret = (yr_pnl / cum_cap) * 100
            yr_wr = len(yr_trades[yr_trades['pnl'] > 0]) / len(yr_trades) * 100
            
            marker = ""
            if year == 2021:
                marker = " ← Target: positive"
            elif year == 2022:
                marker = " ← Target: +15-25%"
            
            print(f"{year:<8} {len(yr_trades):<8} {yr_wr:<8.1f} ${yr_pnl:<14,.0f} {yr_ret:<10.1f}{marker}")
            cum_cap += yr_pnl
        
        # Comparison table
        print("\n" + "="*80)
        print("STRATEGY EVOLUTION:")
        print("="*80)
        print(f"{'Metric':<25} {'V4.6':<15} {'V4.7':<15} {'Change':<15}")
        print("-"*70)
        print(f"{'Bear positions':<25} {'2':<15} {'4':<15} {'+100%':<15}")
        print(f"{'Bear allocation':<25} {'50% max':<15} {'80% max':<15} {'+60%':<15}")
        print(f"{'Bear symbols':<25} {'VXX only':<15} {'VXX+SQQQ+SPXU':<15} {'+2 tools':<15}")
        print(f"{'Quality filter':<25} {'Yes':<15} {'No':<15} {'More trades':<15}")
        print(f"{'Regime confirm':<25} {'5 days':<15} {'3 days':<15} {'Faster':<15}")
        print(f"{'CAGR':<25} {'89.8%':<15} {f'{cagr:.1f}%':<15}")
        print(f"{'Max DD':<25} {'-57.8%':<15} {f'{max_dd:.1f}%':<15}")
        print(f"{'2022 return':<25} {'-21.8%':<15}")
        
        # Find 2022 return
        if 2022 in trades_df['year'].unique():
            yr22_trades = trades_df[trades_df['year'] == 2022]
            yr22_pnl = yr22_trades['pnl'].sum()
            yr22_ret = (yr22_pnl / self.initial_capital) * 100  # Approximate
            print(f"{'2022 return':<25} {'-21.8%':<15} {f'{yr22_ret:.1f}%':<15}")
        
        print("\n✅ V4.7 KEY IMPROVEMENTS:")
        print("  • 4 bear positions (doubled from V4.6)")
        print("  • 80% bear allocation in BEAR regime")
        print("  • NO quality filter on bear trades")
        print("  • 3 bear symbols (VXX, SQQQ, SPXU)")
        print("  • Faster regime detection (3 days)")
        print("  • Emergency crash protection mode")
        print("="*80)

def run_enhanced_v47():
    """Run enhanced V4.7 strategy"""
    strategy = EnhancedDynamicStrategy(
        bull_symbols=["NVDA", "TSLA", "PLTR", "AMD", "COIN"],
        bear_symbols=["VXX", "SQQQ", "SPXU"],
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
    trades, data = run_enhanced_v47()