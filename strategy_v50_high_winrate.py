"""
High Win Rate Momentum Strategy V5.0
═══════════════════════════════════════════════════════════════════════════════

GOAL: 60%+ Win Rate While Maintaining High Returns
  Current V4.8: 34.7% win rate, 203% CAGR
  Target V5.0: 60%+ win rate, 150%+ CAGR

PROBLEM WITH V4.8:
  - Takes too many marginal setups
  - Gets chopped out in sideways markets
  - Stops too tight relative to volatility
  - No confirmation requirements
  - Trades through earnings/events

SOLUTION - 7 ADVANCED FILTERS:
  1. Multi-timeframe alignment (daily + weekly trends agree)
  2. Strong momentum confirmation (not just any momentum)
  3. ADX trending filter (avoid choppy markets)
  4. Volume surge requirement (institutional buying)
  5. Quality score threshold (only grade A setups)
  6. Market regime filter (only trade with SPY trend)
  7. Wider adaptive stops (based on ATR, not fixed %)

Expected: Higher win rate, fewer trades, similar/better total returns
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from pathlib import Path
warnings.filterwarnings('ignore')

class HighWinRateStrategy:
    """V5.0 - Optimized for 60%+ win rate"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
        output_dir: str = "trading_results"
    ):
        self.symbols = symbols or [
            "NVDA", "TSLA", "PLTR", "AMD", "COIN",
            "SMCI", "MSTR", "CRWD", "SNOW", "NET",
            "RIOT", "MARA", "DDOG", "ZS", "MDB"
        ]
        
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("HIGH WIN RATE MOMENTUM STRATEGY V5.0")
        print("Quality over Quantity - Only the Best Setups")
        print("="*80)
        
        # Technical indicators
        self.fast_ma = 10
        self.slow_ma = 30
        self.trend_ma = 50
        self.long_ma = 200
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        self.adx_period = 14
        self.atr_period = 14
        
        # Position management
        self.base_risk = 0.020
        self.quality_mult = 1.5
        self.momentum_mult = 2.0
        self.max_hold = 45
        self.max_positions = 6  # Fewer positions for higher quality
        
        # === NEW: HIGH WIN RATE FILTERS ===
        print("\n🎯 HIGH WIN RATE FILTERS:")
        
        # 1. Multi-timeframe alignment
        self.require_weekly_trend = True
        print("  1. Multi-Timeframe: Require daily + weekly trend alignment")
        
        # 2. Strong momentum threshold
        self.min_momentum_pct = 0.08  # 8% momentum minimum (vs 5% in V4.8)
        print(f"  2. Strong Momentum: Min {self.min_momentum_pct*100}% recent gain")
        
        # 3. ADX trending filter
        self.min_adx = 25  # Only trade in trending markets
        print(f"  3. ADX Filter: Only trade when ADX > {self.min_adx} (trending)")
        
        # 4. Volume surge
        self.min_volume_mult = 1.5  # Require 1.5x avg volume (vs 1.2x)
        print(f"  4. Volume Surge: Require {self.min_volume_mult}x average volume")
        
        # 5. Quality threshold
        self.min_quality = 2  # Only take quality 2+ setups (skip quality 1)
        print(f"  5. Quality Filter: Only take grade {self.min_quality}+ setups")
        
        # 6. Market regime (SPY filter)
        self.require_spy_uptrend = True
        self.spy_data = None
        print("  6. Market Regime: Only trade when SPY is in uptrend")
        
        # 7. Adaptive stops (ATR-based)
        self.atr_stop_multiplier = 2.0  # 2x ATR stop (vs fixed 1.5%)
        self.min_stop_pct = 0.02  # Minimum 2% stop
        self.max_stop_pct = 0.05  # Maximum 5% stop
        print(f"  7. Adaptive Stops: {self.atr_stop_multiplier}x ATR (2-5% range)")
        
        # Take profit adjustment
        self.take_profit = 0.12  # Wider target (12% vs 8%)
        print(f"\n💰 Position Management:")
        print(f"  Take Profit: {self.take_profit*100}% (wider target)")
        print(f"  Max Positions: {self.max_positions} (focus on quality)")
        print(f"  Max Hold: {self.max_hold} days")
        
        self.options_multiplier = 100
        
        print("\n" + "="*80)
        print("TARGET: 60%+ win rate with 150%+ CAGR")
        print("  Strategy: Fewer trades, higher quality, better timing")
        print("  Trade-off: Accept fewer trades for much higher win rate")
        print("="*80 + "\n")
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed TR
        atr = tr.rolling(self.adx_period).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.adx_period).mean()
        
        return adx
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(self.atr_period).mean()
        return atr
    
    def load_spy_data(self) -> pd.DataFrame:
        """Load SPY data for market regime filter"""
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            extended_start = start - pd.Timedelta(days=250)
            
            spy = yf.Ticker("SPY")
            df = spy.history(start=extended_start, end=end, interval='1d')
            
            if df.empty:
                return None
            
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Calculate SPY trend
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            df['spy_uptrend'] = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])
            
            return df[start:end]
        except Exception as e:
            print(f"  Warning: Could not load SPY data: {e}")
            return None
    
    def load_single_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data with enhanced indicators"""
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
            
            # Weekly trend (for multi-timeframe)
            df['sma_weekly'] = df['close'].rolling(50).mean()  # ~10 week MA
            df['weekly_uptrend'] = df['close'] > df['sma_weekly']
            
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
            
            # ADX for trend strength
            df['adx'] = self.calculate_adx(df)
            
            # ATR for adaptive stops
            df['atr'] = self.calculate_atr(df)
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['high_volume'] = df['volume_ratio'] > self.min_volume_mult
            df['extreme_volume'] = df['volume_ratio'] > 2.0
            
            # Price action
            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['higher_low'] = df['low'] > df['low'].shift(1)
            
            # Momentum (stronger threshold)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['momentum_20'] = df['close'].pct_change(20)
            df['strong_momentum'] = df['momentum_10'] > self.min_momentum_pct
            df['extreme_momentum'] = df['momentum_10'] > 0.15
            
            # Trends
            df['uptrend'] = (df['sma_fast'] > df['sma_slow']) & (df['sma_slow'] > df['sma_trend'])
            df['downtrend'] = (df['sma_fast'] < df['sma_slow']) & (df['sma_slow'] < df['sma_trend'])
            df['perfect_uptrend'] = df['uptrend'] & (df['close'] > df['sma_fast'])
            
            # Multi-timeframe alignment
            df['timeframe_aligned'] = df['uptrend'] & df['weekly_uptrend']
            
            # Market regime
            df['bear_market'] = df['close'] < df['sma_trend'] * 0.92
            
            df = df[start:end]
            return df
            
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all symbols + SPY"""
        print(f"Loading data for {len(self.symbols)} symbols + SPY...")
        print(f"Period: {self.start_date} to {self.end_date}\n")
        
        # Load SPY first
        print("Loading SPY (market regime filter)...", end=" ")
        self.spy_data = self.load_spy_data()
        if self.spy_data is not None:
            print(f"✓ {len(self.spy_data)} days")
        else:
            print("✗ Failed (will skip SPY filter)")
        
        all_data = {}
        
        for symbol in self.symbols:
            print(f"Loading {symbol}...", end=" ")
            df = self.load_single_symbol(symbol)
            if df is not None and len(df) > 0:
                all_data[symbol] = df
                print(f"✓ {len(df)} days")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Successfully loaded {len(all_data)}/{len(self.symbols)} symbols\n")
        return all_data
    
    def generate_signals(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate high-quality signals with strict filters"""
        print("="*80)
        print("GENERATING HIGH-QUALITY SIGNALS (STRICT FILTERS)")
        print("="*80)
        
        for symbol, df in all_data.items():
            # Base V4 strategies with ENHANCED filters
            
            # Strategy 1: Classic Trend (ENHANCED)
            trend_long = (
                df['timeframe_aligned'] &  # NEW: Multi-timeframe
                (df['close'] > df['sma_trend']) &
                (df['rsi'] > 45) & (df['rsi'] < 65) &  # Tighter RSI range
                df['high_volume'] &  # NEW: Volume required
                (df['adx'] > self.min_adx) &  # NEW: Trending only
                (~df['bear_market'])
            )
            
            # Strategy 2: Quality Pullback (ENHANCED)
            pullback_long = (
                df['perfect_uptrend'] &
                df['weekly_uptrend'] &  # NEW: Weekly confirmation
                (df['close'] < df['sma_fast']) &
                (df['close'] > df['sma_slow']) &
                (df['rsi'] < 45) & (df['rsi'] > 35) &
                df['higher_low'] &
                df['high_volume'] &  # NEW: Volume surge
                (df['adx'] > self.min_adx)  # NEW: Trending
            )
            
            # Strategy 3: Breakout (ENHANCED)
            breakout_long = (
                (df['close'] > df['high'].shift(1)) &
                (df['close'] > df['bb_upper']) &
                df['extreme_volume'] &  # NEW: Extreme volume required
                df['perfect_uptrend'] &
                df['strong_momentum'] &
                (df['adx'] > self.min_adx + 5)  # NEW: Very strong trend
            )
            
            # Strategy 4: Momentum Acceleration (ENHANCED)
            momentum_long = (
                df['extreme_momentum'] &
                df['timeframe_aligned'] &  # NEW: Multi-timeframe
                (df['rsi'] > 55) & (df['rsi'] < 70) &  # Higher RSI OK for momentum
                df['extreme_volume'] &  # NEW: Must have volume
                (df['momentum_20'] > 0.15) &  # Higher threshold
                (df['adx'] > self.min_adx)
            )
            
            # Strategy 5: Oversold Bounce (ENHANCED - STRICT)
            bounce_long = (
                (df['close'] > df['sma_trend'] * 0.99) &  # Must be close to trend
                (df['rsi'] < 25) &  # More oversold
                (df['close'] < df['bb_lower']) &
                df['weekly_uptrend'] &  # NEW: Weekly must be up
                df['high_volume'] &  # NEW: Bounce needs volume
                (~df['bear_market'])
            )
            
            # Strategy 6: Early Recovery (ENHANCED)
            recovery_long = (
                df['bear_market'] &
                (df['sma_fast'] > df['sma_slow']) &
                (df['rsi'] > 55) &  # Stronger RSI
                (df['close'] > df['close'].shift(1)) &
                df['extreme_volume'] &  # NEW: Need strong volume for recovery
                df['strong_momentum']  # NEW: Need momentum confirmation
            )
            
            df['signal'] = 0
            df['signal_type'] = 'none'
            
            df.loc[trend_long, ['signal', 'signal_type']] = [1, 'trend']
            df.loc[pullback_long, ['signal', 'signal_type']] = [2, 'pullback']
            df.loc[breakout_long, ['signal', 'signal_type']] = [3, 'breakout']
            df.loc[momentum_long, ['signal', 'signal_type']] = [4, 'momentum']
            df.loc[bounce_long, ['signal', 'signal_type']] = [5, 'bounce']
            df.loc[recovery_long, ['signal', 'signal_type']] = [6, 'recovery']
            
            # Exit signal (unchanged)
            df['exit_signal'] = (
                (df['sma_fast'] < df['sma_slow']) |
                (df['close'] < df['sma_trend'] * 0.97) |
                (df['adx'] < 20)  # NEW: Exit when trend weakens
            )
            
            # Enhanced quality scoring
            df['quality'] = 1
            
            # Quality 2: Good setups
            high_quality = (
                (df['signal'] > 0) &
                df['timeframe_aligned'] &
                df['high_volume'] &
                (df['momentum_10'] > 0.05) &
                (df['adx'] > self.min_adx)
            )
            df.loc[high_quality, 'quality'] = 2
            
            # Quality 3: Exceptional setups
            ultra_quality = (
                (df['signal'].isin([3, 4])) |  # Breakout or momentum
                ((df['signal'] > 0) & df['extreme_momentum'] & df['extreme_volume'])
            )
            df.loc[ultra_quality, 'quality'] = 3
            
            all_data[symbol] = df
            
            # Count signals that pass quality filter
            quality_signals = (df['signal'] > 0) & (df['quality'] >= self.min_quality)
            total_signals = quality_signals.sum()
            print(f"{symbol}: {total_signals} high-quality signals (Q{self.min_quality}+)")
        
        print()
        return all_data
    
    def backtest(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run backtest with high win rate filters"""
        print("="*80)
        print("RUNNING HIGH WIN RATE BACKTEST")
        print("="*80 + "\n")
        
        trades = []
        positions = {}
        capital = self.initial_capital
        
        all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        equity_curve = []
        
        for current_date in all_dates:
            # Get current prices
            current_prices = {}
            for symbol in self.symbols:
                if symbol in all_data and current_date in all_data[symbol].index:
                    current_prices[symbol] = all_data[symbol].loc[current_date]
            
            # Check SPY regime if available
            spy_uptrend = True
            if self.spy_data is not None and self.require_spy_uptrend:
                if current_date in self.spy_data.index:
                    spy_uptrend = self.spy_data.loc[current_date, 'spy_uptrend']
            
            # === MANAGE EXISTING POSITIONS ===
            symbols_to_remove = []
            
            for symbol, pos in list(positions.items()):
                if symbol not in current_prices:
                    continue
                
                row = current_prices[symbol]
                current_price = row['close']
                days_held = (current_date - pos['entry_date']).days
                profit_pct = (current_price - pos['entry_price']) / pos['entry_price']
                
                exit_trade = False
                exit_reason = ""
                
                if current_price <= pos['stop_loss']:
                    exit_trade, exit_reason = True, "Stop Loss"
                elif current_price >= pos['take_profit']:
                    exit_trade, exit_reason = True, "Take Profit"
                elif row.get('exit_signal', False):
                    exit_trade, exit_reason = True, "Exit Signal"
                elif days_held > self.max_hold:
                    exit_trade, exit_reason = True, "Time Exit"
                
                if exit_trade:
                    pnl = (current_price - pos['entry_price']) * pos['shares'] * self.options_multiplier
                    capital += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'return_pct': profit_pct * 100,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'signal_type': pos['signal_type'],
                        'quality': pos['quality']
                    })
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del positions[symbol]
            
            # === NEW ENTRIES (with strict filters) ===
            if len(positions) < self.max_positions and spy_uptrend:
                opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol in positions:
                        continue
                    
                    signal = row.get('signal', 0)
                    quality = row.get('quality', 1)
                    
                    # STRICT FILTERS
                    if signal > 0 and quality >= self.min_quality:
                        opportunities.append({
                            'symbol': symbol,
                            'quality': quality,
                            'price': row['close'],
                            'signal_type': row.get('signal_type', 'unknown'),
                            'atr': row.get('atr', row['close'] * 0.02)
                        })
                
                # Sort by quality (highest first)
                opportunities.sort(key=lambda x: x['quality'], reverse=True)
                
                for opp in opportunities[:self.max_positions - len(positions)]:
                    symbol = opp['symbol']
                    current_price = opp['price']
                    atr = opp['atr']
                    
                    # Quality-based position sizing
                    base_risk = self.base_risk
                    if opp['quality'] == 2:
                        base_risk *= self.quality_mult
                    elif opp['quality'] == 3:
                        base_risk *= self.momentum_mult
                    
                    # ADAPTIVE STOP (ATR-based)
                    stop_distance = self.atr_stop_multiplier * atr
                    stop_pct = stop_distance / current_price
                    stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_pct))
                    
                    stop_loss = current_price * (1 - stop_pct)
                    take_profit = current_price * (1 + self.take_profit)
                    
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
                            'signal_type': opp['signal_type'],
                            'quality': opp['quality'],
                            'stop_pct': stop_pct
                        }
            
            # Track equity
            unrealized_pnl = sum(
                (current_prices[sym]['close'] - pos['entry_price']) * pos['shares'] * self.options_multiplier
                for sym, pos in positions.items() if sym in current_prices
            )
            equity_curve.append(capital + unrealized_pnl)
        
        # Add to dataframe
        if all_data:
            first_sym = list(all_data.keys())[0]
            all_data[first_sym]['portfolio_equity'] = pd.Series(equity_curve, index=all_dates)
        
        return trades, all_data
    
    def save_trades_csv(self, trades: List[Dict]):
        """Save trades to CSV"""
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        csv_path = self.output_dir / f"trades_v50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"✓ Trades saved to: {csv_path}")
        return csv_path
    
    def plot_equity_curve(self, all_data: Dict[str, pd.DataFrame], trades: List[Dict]):
        """Plot equity curve with enhanced formatting"""
        first_sym = list(all_data.keys())[0]
        if 'portfolio_equity' not in all_data[first_sym].columns:
            return
        
        equity = all_data[first_sym]['portfolio_equity'].dropna()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('V5.0 High Win Rate Strategy', fontsize=16, fontweight='bold')
        
        # Plot equity
        ax1.plot(equity.index, equity.values, linewidth=2, label='Portfolio Value', color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        
        # Dollar formatting
        def dollar_formatter(x, p):
            if x >= 1e6:
                return f'${x/1e6:.1f}M'
            elif x >= 1000:
                return f'${x/1000:.0f}K'
            else:
                return f'${x:.0f}'
        ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
        
        # Annotations
        max_equity = equity.max()
        final_equity = equity.iloc[-1]
        
        ax1.annotate(f'Peak: {dollar_formatter(max_equity, None)}', 
                    xy=(equity.idxmax(), max_equity),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='darkgreen'))
        
        ax1.annotate(f'Final: {dollar_formatter(final_equity, None)}', 
                    xy=(equity.index[-1], final_equity),
                    xytext=(-80, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='darkblue'))
        
        ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Portfolio Growth Over Time', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Portfolio Drawdown', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f"equity_curve_v50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Equity curve saved to: {plot_path}")
        plt.show()
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Analyze with focus on win rate improvement"""
        print("\n" + "="*80)
        print("HIGH WIN RATE STRATEGY V5.0 RESULTS")
        print("="*80)
        
        if not trades:
            print("\nNo trades executed.")
            return
        
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['year'] = trades_df['entry_date'].dt.year
        
        total_pnl = trades_df['pnl'].sum()
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winners) / len(trades_df) * 100
        
        # Calculate metrics
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
        
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"  Period: {years:.2f} years")
        print(f"  Initial: ${self.initial_capital:,.0f}")
        print(f"  Final: ${self.initial_capital + total_pnl:,.0f}")
        print(f"  Total Return: {(total_pnl/self.initial_capital*100):.1f}%")
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Max DD: {max_dd:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        
        print(f"\n🎯 WIN RATE ANALYSIS:")
        print(f"  Win Rate: {win_rate:.1f}% {'✅ TARGET MET!' if win_rate >= 60 else '⚠️ Below target'}")
        print(f"  Winners: {len(winners)} trades")
        print(f"  Losers: {len(losers)} trades")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Avg Winner: ${winners['pnl'].mean():,.0f}")
        print(f"  Avg Loser: ${losers['pnl'].mean():,.0f}")
        print(f"  Win/Loss Ratio: {abs(winners['pnl'].mean() / losers['pnl'].mean()):.2f}")
        
        # Compare to V4.8
        print("\n" + "="*80)
        print("STRATEGY COMPARISON:")
        print("="*80)
        print(f"{'Metric':<25} {'V4.8':<15} {'V5.0':<15} {'Change':<15}")
        print("-"*70)
        print(f"{'Win Rate':<25} {'34.7%':<15} {f'{win_rate:.1f}%':<15} {f'+{win_rate-34.7:.1f}%':<15}")
        print(f"{'CAGR':<25} {'203.2%':<15} {f'{cagr:.1f}%':<15} {f'{cagr-203.2:+.1f}%':<15}")
        print(f"{'Max DD':<25} {'-90.1%':<15} {f'{max_dd:.1f}%':<15} {f'{max_dd+90.1:+.1f}%':<15}")
        print(f"{'Total Trades':<25} {'2,254':<15} {f'{len(trades_df):,}':<15} {f'{len(trades_df)-2254:+,}':<15}")
        print(f"{'Sharpe':<25} {'1.48':<15} {f'{sharpe:.2f}':<15} {f'{sharpe-1.48:+.2f}':<15}")
        
        print("\n✅ V5.0 IMPROVEMENTS:")
        if win_rate >= 60:
            print("  • 60%+ win rate ACHIEVED! 🎯")
        print(f"  • Stricter entry filters = higher quality")
        print(f"  • Adaptive ATR stops = fewer false stops")
        print(f"  • Multi-timeframe = better timing")
        print(f"  • Volume confirmation = institutional support")
        print(f"  • Market regime filter = avoid bad periods")
        print("="*80)

def run_v50():
    """Run V5.0 high win rate strategy"""
    strategy = HighWinRateStrategy(
        start_date="2020-01-01",
        initial_capital=20000
    )
    
    all_data = strategy.load_all_data()
    if not all_data:
        print("\n❌ Failed to load data")
        return None, None
    
    all_data = strategy.generate_signals(all_data)
    trades, all_data = strategy.backtest(all_data)
    
    strategy.save_trades_csv(trades)
    strategy.plot_equity_curve(all_data, trades)
    strategy.analyze_results(trades, all_data)
    
    return trades, all_data

if __name__ == "__main__":
    trades, data = run_v50()
