"""
Advanced Risk-Managed Momentum Strategy V4.9
═══════════════════════════════════════════════════════════════════════════════

NEW: Advanced Drawdown Protection Tools
  ✓ Dynamic position sizing based on drawdown
  ✓ Volatility-adjusted risk
  ✓ Correlation-aware position limits
  ✓ Portfolio heat management
  ✓ Trailing equity stops
  ✓ Enhanced charts with dollar values
  
GOAL: Smoother equity curve closer to monotonic growth
  - V4: 97% CAGR, -79% DD ❌
  - V4.8: ~90% CAGR, -50-60% DD 😐
  - V4.9: 70-80% CAGR, -30-40% DD ✅ (TARGET)

ADVANCED RISK TOOLS:
  1. Drawdown Scaling: Reduce size when portfolio is down
  2. Volatility Adjustment: Scale based on VIX/market vol
  3. Correlation Filter: Avoid too many correlated positions
  4. Portfolio Heat: Max 15% total portfolio risk at once
  5. Trailing Equity Stop: Cut all positions if down 20%
  6. Win/Loss Streak Adjustment: Reduce after losses
  
Expected: Smoother growth, lower DD, better Sharpe
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

class AdvancedRiskManagedStrategy:
    """V4.9 - Advanced risk management for smoother returns"""
    
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
        
        # Technical indicators
        self.fast_ma = 10
        self.slow_ma = 30
        self.trend_ma = 50
        self.long_ma = 200
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2
        
        print("\n" + "="*80)
        print("ADVANCED RISK-MANAGED MOMENTUM STRATEGY V4.9")
        print("Smoother equity curve through dynamic risk management")
        print("="*80)
        
        # Base V4 parameters
        self.base_risk = 0.020              # 2% base
        self.quality_mult = 1.5             # 3% on quality
        self.momentum_mult = 2.0            # 4% on ultra momentum
        self.stop_loss = 0.015              # 1.5%
        self.take_profit = 0.08             # 8%
        self.max_hold = 45
        self.max_positions = 8
        
        # === ADVANCED RISK MANAGEMENT ===
        print("\n🛡️ ADVANCED RISK CONTROLS:")
        
        # 1. Drawdown-based scaling
        self.drawdown_thresholds = {
            0.00: 1.00,   # 0-10% DD: 100% size
            0.10: 0.75,   # 10-20% DD: 75% size
            0.20: 0.50,   # 20-30% DD: 50% size
            0.30: 0.25,   # 30-40% DD: 25% size
            0.40: 0.10    # 40%+ DD: 10% size (survival mode)
        }
        print("  1. Drawdown Scaling: Reduce position size in drawdowns")
        for dd, scale in self.drawdown_thresholds.items():
            print(f"     {dd*100:.0f}%+ DD → {scale*100:.0f}% position size")
        
        # 2. Volatility adjustment
        self.vol_target = 25              # Target 25% annualized vol
        self.vol_lookback = 20            # 20-day vol window
        self.max_vol_adjustment = 0.5     # Don't go below 50% size
        print(f"\n  2. Volatility Scaling: Target {self.vol_target}% annualized vol")
        print(f"     Reduces size in high volatility environments")
        
        # 3. Correlation filter
        self.max_correlation = 0.70       # Don't take positions >70% correlated
        self.correlation_lookback = 60    # 60-day correlation
        print(f"\n  3. Correlation Filter: Max {self.max_correlation:.0%} correlation")
        print(f"     Avoids clustered positions in same sector")
        
        # 4. Portfolio heat
        self.max_portfolio_heat = 0.15    # Max 15% total risk
        print(f"\n  4. Portfolio Heat: Max {self.max_portfolio_heat*100}% total risk")
        print(f"     Limits total exposure across all positions")
        
        # 5. Trailing equity stop
        self.trailing_equity_stop = 0.20  # Stop all if down 20% from peak
        self.peak_equity = initial_capital
        self.equity_stop_active = False
        print(f"\n  5. Trailing Equity Stop: Exit all if down {self.trailing_equity_stop*100}% from peak")
        print(f"     Emergency protection against severe drawdowns")
        
        # 6. Streak adjustment
        self.max_consecutive_losses = 3   # Reduce after 3 losses
        self.loss_streak_multiplier = 0.5 # Cut size in half after streak
        self.consecutive_losses = 0
        print(f"\n  6. Loss Streak Protection: Reduce size after {self.max_consecutive_losses} losses")
        
        self.options_multiplier = 100
        
        print("\n" + "="*80)
        print("GOAL: Smoother, more consistent returns")
        print("  Target CAGR: 70-80% (lower but safer)")
        print("  Target Max DD: -30-40% (much better)")
        print("  Target Sharpe: 1.5+ (better risk-adjusted)")
        print("="*80 + "\n")
    
    def get_drawdown_multiplier(self, current_equity: float) -> float:
        """Calculate position size multiplier based on current drawdown"""
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        drawdown = abs(min(0, drawdown))  # Convert to positive DD
        
        # Find appropriate multiplier
        for threshold, multiplier in sorted(self.drawdown_thresholds.items(), reverse=True):
            if drawdown >= threshold:
                return multiplier
        return 1.0
    
    def get_volatility_multiplier(self, symbol: str, date: datetime, all_data: Dict) -> float:
        """Calculate position size multiplier based on recent volatility"""
        if symbol not in all_data or date not in all_data[symbol].index:
            return 1.0
        
        df = all_data[symbol]
        if date not in df.index:
            return 1.0
        
        # Calculate recent volatility
        recent_data = df.loc[:date].tail(self.vol_lookback)
        if len(recent_data) < self.vol_lookback:
            return 1.0
        
        returns = recent_data['close'].pct_change().dropna()
        current_vol = returns.std() * np.sqrt(252) * 100
        
        if current_vol == 0:
            return 1.0
        
        # Scale position size inversely with volatility
        vol_multiplier = self.vol_target / current_vol
        
        # Don't go too extreme
        vol_multiplier = max(self.max_vol_adjustment, min(1.5, vol_multiplier))
        
        return vol_multiplier
    
    def check_correlation(self, symbol: str, date: datetime, positions: Dict, all_data: Dict) -> bool:
        """Check if new position is too correlated with existing positions"""
        if not positions or symbol not in all_data:
            return True  # OK to enter
        
        symbol_data = all_data[symbol]
        if date not in symbol_data.index:
            return True
        
        # Get recent returns for new symbol
        symbol_recent = symbol_data.loc[:date].tail(self.correlation_lookback)
        if len(symbol_recent) < 30:  # Need minimum data
            return True
        
        symbol_returns = symbol_recent['close'].pct_change().dropna()
        
        # Check correlation with each existing position
        for pos_symbol in positions.keys():
            if pos_symbol not in all_data:
                continue
            
            pos_data = all_data[pos_symbol]
            pos_recent = pos_data.loc[:date].tail(self.correlation_lookback)
            if len(pos_recent) < 30:
                continue
            
            pos_returns = pos_recent['close'].pct_change().dropna()
            
            # Align dates
            common_dates = symbol_returns.index.intersection(pos_returns.index)
            if len(common_dates) < 20:
                continue
            
            corr = symbol_returns.loc[common_dates].corr(pos_returns.loc[common_dates])
            
            if abs(corr) > self.max_correlation:
                return False  # Too correlated, reject
        
        return True  # OK to enter
    
    def calculate_portfolio_heat(self, positions: Dict, current_prices: Dict) -> float:
        """Calculate current total portfolio risk"""
        total_risk = 0
        for symbol, pos in positions.items():
            if symbol in current_prices:
                position_risk = (pos['entry_price'] - pos['stop_loss']) * pos['shares'] * self.options_multiplier
                total_risk += position_risk
        return total_risk
    
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
            
            # Market regime
            df['bear_market'] = df['close'] < df['sma_trend'] * 0.92
            
            df = df[start:end]
            return df
            
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all symbols"""
        print(f"Loading data for {len(self.symbols)} symbols...")
        print(f"Period: {self.start_date} to {self.end_date}\n")
        
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
        """Generate V4 momentum signals"""
        print("="*80)
        print("GENERATING V4 MOMENTUM SIGNALS")
        print("="*80)
        
        for symbol, df in all_data.items():
            # V4 strategies (same as before)
            trend_long = (
                (df['sma_fast'] > df['sma_slow']) &
                (df['close'] > df['sma_trend']) &
                (df['rsi'] > 40) & (df['rsi'] < 70) &
                (~df['bear_market'])
            )
            
            pullback_long = (
                df['perfect_uptrend'] &
                (df['close'] < df['sma_fast']) &
                (df['close'] > df['sma_slow']) &
                (df['rsi'] < 50) & (df['rsi'] > 35) &
                df['higher_low']
            )
            
            breakout_long = (
                (df['close'] > df['high'].shift(1)) &
                (df['close'] > df['bb_upper']) &
                (df['volume'] > df['volume_sma'] * 1.5) &
                df['perfect_uptrend'] &
                df['strong_momentum']
            )
            
            momentum_long = (
                df['extreme_momentum'] &
                df['perfect_uptrend'] &
                (df['rsi'] > 50) & (df['rsi'] < 65) &
                df['high_volume'] &
                (df['momentum_20'] > 0.10)
            )
            
            bounce_long = (
                (df['close'] > df['sma_trend'] * 0.97) &
                (df['rsi'] < 30) &
                (df['close'] < df['bb_lower']) &
                (~df['bear_market'])
            )
            
            recovery_long = (
                df['bear_market'] &
                (df['sma_fast'] > df['sma_slow']) &
                (df['rsi'] > 50) &
                (df['close'] > df['close'].shift(1)) &
                df['high_volume']
            )
            
            df['signal'] = 0
            df['signal_type'] = 'none'
            
            df.loc[trend_long, ['signal', 'signal_type']] = [1, 'trend']
            df.loc[pullback_long, ['signal', 'signal_type']] = [2, 'pullback']
            df.loc[breakout_long, ['signal', 'signal_type']] = [3, 'breakout']
            df.loc[momentum_long, ['signal', 'signal_type']] = [4, 'momentum']
            df.loc[bounce_long, ['signal', 'signal_type']] = [5, 'bounce']
            df.loc[recovery_long, ['signal', 'signal_type']] = [6, 'recovery']
            
            df['exit_signal'] = (
                (df['sma_fast'] < df['sma_slow']) |
                (df['close'] < df['sma_trend'] * 0.97)
            )
            
            df['quality'] = 1
            high_quality = (
                (df['signal'] > 0) &
                df['perfect_uptrend'] &
                df['high_volume'] &
                (df['momentum_10'] > 0.03)
            )
            df.loc[high_quality, 'quality'] = 2
            
            ultra_quality = (
                (df['signal'].isin([4])) |
                ((df['signal'] > 0) & df['extreme_momentum'])
            )
            df.loc[ultra_quality, 'quality'] = 3
            
            all_data[symbol] = df
            
            total_signals = (df['signal'] > 0).sum()
            print(f"{symbol}: {total_signals} signals")
        
        print()
        return all_data
    
    def backtest(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Run backtest with advanced risk management"""
        print("="*80)
        print("RUNNING BACKTEST WITH ADVANCED RISK MANAGEMENT")
        print("="*80 + "\n")
        
        trades = []
        positions = {}
        capital = self.initial_capital
        self.peak_equity = self.initial_capital
        
        all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        equity_curve = []
        risk_metrics = []
        
        for current_date in all_dates:
            # Get current prices
            current_prices = {}
            for symbol in self.symbols:
                if symbol in all_data and current_date in all_data[symbol].index:
                    current_prices[symbol] = all_data[symbol].loc[current_date]
            
            # Calculate current equity
            unrealized_pnl = sum(
                (current_prices[sym]['close'] - pos['entry_price']) * pos['shares'] * self.options_multiplier
                for sym, pos in positions.items() if sym in current_prices
            )
            current_equity = capital + unrealized_pnl
            
            # Update peak and check trailing stop
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                self.equity_stop_active = False
            
            drawdown_from_peak = (current_equity - self.peak_equity) / self.peak_equity
            if drawdown_from_peak < -self.trailing_equity_stop:
                if not self.equity_stop_active:
                    print(f"\n🚨 TRAILING EQUITY STOP TRIGGERED on {current_date.date()}")
                    print(f"   Down {drawdown_from_peak*100:.1f}% from peak of ${self.peak_equity:,.0f}")
                    print(f"   Exiting all positions")
                    self.equity_stop_active = True
            
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
                
                # Force exit if equity stop active
                if self.equity_stop_active:
                    exit_trade, exit_reason = True, "Equity Stop"
                elif current_price <= pos['stop_loss']:
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
                    
                    # Track win/loss streak
                    if pnl > 0:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                    
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
            
            # === NEW ENTRIES (if not in equity stop mode) ===
            if not self.equity_stop_active and len(positions) < self.max_positions:
                # Calculate current portfolio heat
                current_heat = self.calculate_portfolio_heat(positions, current_prices)
                
                # Get risk multipliers
                dd_multiplier = self.get_drawdown_multiplier(current_equity)
                
                # Loss streak adjustment
                streak_multiplier = 1.0
                if self.consecutive_losses >= self.max_consecutive_losses:
                    streak_multiplier = self.loss_streak_multiplier
                
                opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol in positions:
                        continue
                    
                    if row.get('signal', 0) > 0:
                        # Check correlation
                        if not self.check_correlation(symbol, current_date, positions, all_data):
                            continue
                        
                        # Get volatility multiplier
                        vol_multiplier = self.get_volatility_multiplier(symbol, current_date, all_data)
                        
                        opportunities.append({
                            'symbol': symbol,
                            'quality': row.get('quality', 1),
                            'price': row['close'],
                            'signal_type': row.get('signal_type', 'unknown'),
                            'vol_multiplier': vol_multiplier
                        })
                
                opportunities.sort(key=lambda x: x['quality'], reverse=True)
                
                for opp in opportunities[:self.max_positions - len(positions)]:
                    symbol = opp['symbol']
                    current_price = opp['price']
                    
                    # Base position sizing
                    base_risk = self.base_risk
                    if opp['quality'] == 2:
                        base_risk *= self.quality_mult
                    elif opp['quality'] == 3:
                        base_risk *= self.momentum_mult
                    
                    # Apply all risk multipliers
                    adjusted_risk = base_risk * dd_multiplier * opp['vol_multiplier'] * streak_multiplier
                    
                    stop_loss = current_price * (1 - self.stop_loss)
                    take_profit = current_price * (1 + self.take_profit)
                    
                    risk_per_share = (current_price - stop_loss) * self.options_multiplier
                    risk_amount = capital * adjusted_risk
                    
                    # Check portfolio heat limit
                    if current_heat + risk_amount > capital * self.max_portfolio_heat:
                        continue  # Skip, would exceed heat limit
                    
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
                            'quality': opp['quality']
                        }
                        current_heat += risk_amount
            
            # Track equity and risk metrics
            equity_curve.append(current_equity)
            risk_metrics.append({
                'date': current_date,
                'equity': current_equity,
                'drawdown_pct': drawdown_from_peak * 100,
                'dd_multiplier': self.get_drawdown_multiplier(current_equity),
                'positions': len(positions),
                'portfolio_heat': current_heat / capital if capital > 0 else 0
            })
        
        # Add to dataframe
        if all_data:
            first_sym = list(all_data.keys())[0]
            all_data[first_sym]['portfolio_equity'] = pd.Series(equity_curve, index=all_dates)
            risk_df = pd.DataFrame(risk_metrics).set_index('date')
            all_data[first_sym] = all_data[first_sym].join(risk_df, how='left')
        
        return trades, all_data
    
    def plot_enhanced_equity_curve(self, all_data: Dict[str, pd.DataFrame], trades: List[Dict]):
        """Plot equity curve with dollar values and risk metrics"""
        first_sym = list(all_data.keys())[0]
        if 'portfolio_equity' not in all_data[first_sym].columns:
            print("No equity data to plot")
            return
        
        equity = all_data[first_sym]['portfolio_equity'].dropna()
        trades_df = pd.DataFrame(trades)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        fig.suptitle('V4.9 Advanced Risk-Managed Strategy', fontsize=16, fontweight='bold')
        
        # === PLOT 1: Equity Curve with Dollar Values ===
        ax1.plot(equity.index, equity.values, linewidth=2, label='Portfolio Value', color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Add dollar value annotations at key points
        max_equity = equity.max()
        max_equity_date = equity.idxmax()
        final_equity = equity.iloc[-1]
        
        # Annotate max equity
        ax1.annotate(f'Peak: ${max_equity:,.0f}', 
                    xy=(max_equity_date, max_equity),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='darkgreen'),
                    fontsize=10, fontweight='bold')
        
        # Annotate final equity
        ax1.annotate(f'Final: ${final_equity:,.0f}', 
                    xy=(equity.index[-1], final_equity),
                    xytext=(-80, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='darkblue'),
                    fontsize=10, fontweight='bold')
        
        # Format y-axis with dollar signs
        def dollar_formatter(x, p):
            return f'${x/1000:.0f}K' if x >= 1000 else f'${x:.0f}'
        ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
        
        # Add trade markers
        if not trades_df.empty:
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            
            winning_entries = trades_df[trades_df['pnl'] > 0]['entry_date']
            losing_entries = trades_df[trades_df['pnl'] <= 0]['entry_date']
            
            for date in winning_entries:
                if date in equity.index:
                    ax1.scatter(date, equity.loc[date], color='green', marker='^', s=40, alpha=0.4, zorder=5)
            
            for date in losing_entries:
                if date in equity.index:
                    ax1.scatter(date, equity.loc[date], color='red', marker='v', s=40, alpha=0.4, zorder=5)
        
        ax1.set_ylabel('Portfolio Value', fontsize=12, fontweight='bold')
        ax1.set_title(f'Portfolio Growth: ${self.initial_capital:,.0f} → ${final_equity:,.0f} ({(final_equity/self.initial_capital-1)*100:.1f}%)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # === PLOT 2: Drawdown ===
        returns = equity.pct_change().dropna()
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                         where=(drawdown.values < 0), color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        
        # Annotate max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax2.annotate(f'Max DD: {max_dd:.1f}%', 
                    xy=(max_dd_date, max_dd),
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='salmon', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='darkred'),
                    fontsize=10, fontweight='bold')
        
        ax2.axhline(y=-20, color='orange', linestyle='--', alpha=0.5, label='Trailing Stop (-20%)')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Portfolio Drawdown', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # === PLOT 3: Risk Metrics ===
        if 'dd_multiplier' in all_data[first_sym].columns:
            dd_mult = all_data[first_sym]['dd_multiplier'].fillna(1.0)
            heat = all_data[first_sym]['portfolio_heat'].fillna(0) * 100
            
            ax3_twin = ax3.twinx()
            
            # Plot position size multiplier
            ax3.plot(dd_mult.index, dd_mult.values, color='blue', linewidth=2, label='Position Size Multiplier')
            ax3.fill_between(dd_mult.index, 0, dd_mult.values, alpha=0.2, color='blue')
            ax3.set_ylabel('Position Size Multiplier', fontsize=11, fontweight='bold', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            ax3.set_ylim(0, 1.2)
            
            # Plot portfolio heat
            ax3_twin.plot(heat.index, heat.values, color='orange', linewidth=2, label='Portfolio Heat')
            ax3_twin.fill_between(heat.index, 0, heat.values, alpha=0.2, color='orange')
            ax3_twin.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Max Heat (15%)')
            ax3_twin.set_ylabel('Portfolio Heat (%)', fontsize=11, fontweight='bold', color='orange')
            ax3_twin.tick_params(axis='y', labelcolor='orange')
            ax3_twin.set_ylim(0, 20)
            
            ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax3.set_title('Dynamic Risk Management', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # Combined legend
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"equity_curve_v49_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Enhanced equity curve saved to: {plot_path}")
        
        plt.show()
    
    def save_trades_csv(self, trades: List[Dict]):
        """Save trades to CSV"""
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        csv_path = self.output_dir / f"trades_v49_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"✓ Trades saved to: {csv_path}")
        return csv_path
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Analyze results with focus on risk management"""
        print("\n" + "="*80)
        print("ADVANCED RISK-MANAGED STRATEGY V4.9 RESULTS")
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
            
            # Calmar ratio
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        else:
            sharpe, max_dd, cagr, years, calmar = 0, 0, 0, 0, 0
        
        print("\n📊 OVERALL PERFORMANCE:")
        print(f"  Period: {years:.2f} years")
        print(f"  Initial: ${self.initial_capital:,.0f}")
        print(f"  Final: ${self.initial_capital + total_pnl:,.0f}")
        print(f"  Total Return: {(total_pnl/self.initial_capital*100):.1f}%")
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Max DD: {max_dd:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Calmar: {calmar:.2f}")
        print(f"  Win Rate: {win_rate:.1f}%")
        
        # Equity stop tracking
        equity_stops = trades_df[trades_df['exit_reason'] == 'Equity Stop']
        if len(equity_stops) > 0:
            print(f"\n🚨 Trailing Equity Stops: {len(equity_stops)} trades exited")
        
        # Strategy comparison
        print("\n" + "="*80)
        print("STRATEGY EVOLUTION:")
        print("="*80)
        print(f"{'Metric':<25} {'V4':<15} {'V4.8':<15} {'V4.9':<15}")
        print("-"*70)
        print(f"{'Approach':<25} {'Simple':<15} {'Diversified':<15} {'Risk-Managed':<15}")
        print(f"{'CAGR':<25} {'97.2%':<15} {'~90%':<15} {f'{cagr:.1f}%':<15}")
        print(f"{'Max DD':<25} {'-79.2%':<15} {'-50-60%':<15} {f'{max_dd:.1f}%':<15}")
        print(f"{'Sharpe':<25} {'1.25':<15} {'~1.3':<15} {f'{sharpe:.2f}':<15}")
        print(f"{'Calmar':<25} {'1.23':<15} {'~1.5':<15} {f'{calmar:.2f}':<15}")
        
        print("\n✅ V4.9 IMPROVEMENTS:")
        print("  • Drawdown-based position scaling")
        print("  • Volatility-adjusted risk")
        print("  • Correlation filtering")
        print("  • Portfolio heat limits (15% max)")
        print("  • Trailing equity stop (-20%)")
        print("  • Loss streak protection")
        print("  • Enhanced charts with $ values")
        print("="*80)

def run_v49():
    """Run V4.9 with advanced risk management"""
    strategy = AdvancedRiskManagedStrategy(
        start_date="2020-01-01",
        initial_capital=10000
    )
    
    all_data = strategy.load_all_data()
    if not all_data:
        print("\n❌ Failed to load data")
        return None, None
    
    all_data = strategy.generate_signals(all_data)
    trades, all_data = strategy.backtest(all_data)
    
    # Save and plot
    strategy.save_trades_csv(trades)
    strategy.plot_enhanced_equity_curve(all_data, trades)
    strategy.analyze_results(trades, all_data)
    
    return trades, all_data

if __name__ == "__main__":
    trades, data = run_v49()
