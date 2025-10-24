"""
Simplified Momentum Strategy V4
- Pure momentum strategies only (NO mean reversion)
- Simple bear market protection (SPY < 200 MA)
- Focus on what works: trend following with good risk management
- Goal: Match V2's bull performance, slightly better bear protection
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class SimplifiedMomentumStrategy:
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
        use_bear_protection: bool = True,
    ):
        # self.symbols = symbols or ["NVDA", "ORCL", "TSLA", "PLTR", "IBM"]
        # self.symbols = symbols or ["NVDA", "TSLA", "PLTR", "AMD", "COIN"]
        self.symbols = symbols or ["NVDA", "PLTR", "AMD", "TSLA"]
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.use_bear_protection = use_bear_protection
        
        # === CORE MOMENTUM PARAMETERS (Keep what works from V2) ===
        self.fast_ma = 10
        self.slow_ma = 30
        self.trend_ma = 50
        self.long_ma = 200
        
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        self.bb_period = 20
        self.bb_std = 2
        
        # === RISK MANAGEMENT ===
        # Normal risk (bull market)
        self.base_risk_pct = 0.02  # 2% base
        self.quality_multiplier = 1.5  # 3% on quality
        self.momentum_multiplier = 2.0  # 4% on strong momentum
        
        # SIMPLE bear protection (not complex like V3)
        self.bear_risk_multiplier = 0.5  # 50% reduction (not 70-90%)
        self.bear_max_positions = 3  # 3 positions (not 1-2)
        
        # Stops and targets
        self.stop_loss_pct = 0.015  # 1.5%
        self.take_profit_pct = 0.08  # 8%
        self.trailing_stop_pct = 0.025  # 2.5%
        
        # BETTER stop management (key to bear markets)
        self.bear_stop_multiplier = 0.8  # 1.2% stop in bear (tighter)
        self.bear_time_exit = 20  # Exit after 20 days in bear (vs 45 normally)
        
        # Volatility adjustment
        self.high_vol_reduction = 0.7  # 30% reduction (not 40%)
        self.volatility_threshold = 65  # Only reduce VERY high vol stocks
        
        # Portfolio
        self.max_leverage = 2.0
        self.leverage_threshold = 0.5
        self.max_positions = 5
        
        self.options_multiplier = 100
        self.monthly_trades = {}
        self.spy_data = None
    
    def load_spy_data(self) -> pd.DataFrame:
        """Load SPY for SIMPLE bear market detection"""
        print("\nLoading SPY for bear market detection...")
        
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            extended_start = start - pd.Timedelta(days=250)
            
            spy = yf.Ticker("SPY")
            df = spy.history(start=extended_start, end=end, interval='1d')
            
            if df.empty:
                print("  ✗ Failed to load SPY - will proceed without filter")
                return None
            
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # SIMPLE bear detection: just 200 MA
            df['sma_200'] = df['close'].rolling(200).mean()
            df['is_bear'] = df['close'] < df['sma_200']
            
            # Also track how far below 200 MA
            df['distance_from_200'] = (df['close'] - df['sma_200']) / df['sma_200']
            
            df = df[start:end]
            
            bear_days = df['is_bear'].sum()
            bear_pct = bear_days / len(df) * 100
            
            print(f"  ✓ SPY loaded: {len(df)} days")
            print(f"  Bear market days: {bear_days} ({bear_pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error loading SPY: {e}")
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all symbols"""
        print(f"\n{'='*80}")
        print(f"SIMPLIFIED MOMENTUM STRATEGY V4")
        print(f"Pure momentum + Simple bear protection")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"{'='*80}")
        
        # Load SPY
        if self.use_bear_protection:
            self.spy_data = self.load_spy_data()
        
        all_data = {}
        
        for symbol in self.symbols:
            print(f"\nLoading {symbol}...", end=" ")
            df = self.load_single_symbol(symbol)
            if df is not None and len(df) > 0:
                # Add SPY bear market flag
                if self.spy_data is not None:
                    df = df.join(self.spy_data[['is_bear', 'distance_from_200']], 
                                how='left', rsuffix='_spy')
                    df['spy_bear'] = df['is_bear'].fillna(False)
                    df['spy_distance'] = df['distance_from_200'].fillna(0)
                else:
                    df['spy_bear'] = False
                    df['spy_distance'] = 0
                
                # Volatility
                returns = df['close'].pct_change().dropna()
                annual_vol = returns.std() * np.sqrt(252) * 100
                df['high_volatility_stock'] = annual_vol > self.volatility_threshold
                
                all_data[symbol] = df
                vol_flag = "⚠️  HIGH VOL" if annual_vol > self.volatility_threshold else ""
                print(f"✓ {len(df)} days | Vol: {annual_vol:.1f}% {vol_flag}")
            else:
                print("✗ Failed")
        
        print(f"\n✓ Successfully loaded {len(all_data)} symbols")
        return all_data
    
    def load_single_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data with momentum indicators only"""
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
            df['sma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
            df['sma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
            df['sma_trend'] = df['close'].rolling(window=self.trend_ma).mean()
            df['sma_long'] = df['close'].rolling(window=self.long_ma).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
            df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['high_volume'] = df['volume'] > df['volume_sma'] * 1.2
            df['extreme_volume'] = df['volume'] > df['volume_sma'] * 1.5
            
            # Price action
            df['higher_high'] = df['high'] > df['high'].shift(1)
            df['higher_low'] = df['low'] > df['low'].shift(1)
            
            # Momentum
            df['momentum_5'] = df['close'].pct_change(periods=5)
            df['momentum_10'] = df['close'].pct_change(periods=10)
            df['momentum_20'] = df['close'].pct_change(periods=20)
            df['strong_momentum'] = df['momentum_10'] > 0.05
            df['extreme_momentum'] = df['momentum_10'] > 0.10
            
            # Market regime (stock-specific)
            df['bear_market'] = df['close'] < df['sma_trend'] * 0.92
            df['bull_market'] = df['close'] > df['sma_long']
            
            # Trend quality
            df['perfect_trend'] = (
                (df['sma_fast'] > df['sma_slow']) & 
                (df['sma_slow'] > df['sma_trend']) &
                (df['close'] > df['sma_fast'])
            )
            
            df = df[start:end]
            return df
            
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            return None
    
    def generate_signals(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate MOMENTUM signals only (V2 style, no mean reversion)"""
        
        print(f"\n{'='*80}")
        print("GENERATING MOMENTUM SIGNALS")
        print(f"{'='*80}")
        
        for symbol, df in all_data.items():
            print(f"\n{symbol}:")
            
            # === MOMENTUM STRATEGIES ONLY (From V2) ===
            
            # Strategy 1: Classic Trend Following
            trend_long = (
                (df['sma_fast'] > df['sma_slow']) &
                (df['close'] > df['sma_trend']) &
                (df['rsi'] > 40) & (df['rsi'] < 70) &
                (~df['bear_market'])
            )
            
            # Strategy 2: Quality Pullback
            pullback_long = (
                (df['perfect_trend']) &
                (df['close'] < df['sma_fast']) &
                (df['close'] > df['sma_slow']) &
                (df['rsi'] < 50) & (df['rsi'] > 35) &
                (df['higher_low'])
            )
            
            # Strategy 3: Breakout
            breakout_long = (
                (df['close'] > df['high'].shift(1)) &
                (df['close'] > df['bb_upper']) &
                (df['volume'] > df['volume_sma'] * 1.5) &
                (df['perfect_trend']) &
                (df['strong_momentum'])
            )
            
            # Strategy 4: Momentum Acceleration
            momentum_long = (
                (df['extreme_momentum']) &
                (df['perfect_trend']) &
                (df['rsi'] > 50) & (df['rsi'] < 65) &
                (df['high_volume']) &
                (df['momentum_20'] > 0.10)
            )
            
            # Strategy 5: Oversold Bounce (kept as momentum, not mean reversion)
            bounce_long = (
                (df['close'] > df['sma_trend'] * 0.97) &
                (df['rsi'] < 30) &
                (df['close'] < df['bb_lower']) &
                (~df['bear_market'])
            )
            
            # Strategy 6: Early Recovery
            recovery_long = (
                (df['bear_market']) &
                (df['sma_fast'] > df['sma_slow']) &
                (df['rsi'] > 50) &
                (df['close'] > df['close'].shift(1)) &
                (df['high_volume'])
            )
            
            # Assign signals
            df['signal'] = 0
            df['signal_type'] = 'none'
            
            df.loc[trend_long, ['signal', 'signal_type']] = [1, 'momentum_trend']
            df.loc[pullback_long, ['signal', 'signal_type']] = [2, 'momentum_pullback']
            df.loc[breakout_long, ['signal', 'signal_type']] = [3, 'momentum_breakout']
            df.loc[momentum_long, ['signal', 'signal_type']] = [4, 'momentum_acceleration']
            df.loc[bounce_long, ['signal', 'signal_type']] = [5, 'momentum_bounce']
            df.loc[recovery_long, ['signal', 'signal_type']] = [6, 'momentum_recovery']
            
            # Exit signals
            df['exit_signal'] = (
                (df['sma_fast'] < df['sma_slow']) |
                (df['close'] < df['sma_trend'] * 0.97)
            )
            
            # Signal quality
            df['signal_quality'] = 0
            df.loc[df['signal'] > 0, 'signal_quality'] = 1
            
            high_quality = (
                (df['signal'] > 0) &
                (df['perfect_trend']) &
                (df['high_volume']) &
                (df['momentum_10'] > 0.03)
            )
            df.loc[high_quality, 'signal_quality'] = 2
            
            ultra_quality = (
                (df['signal'].isin([4])) |
                ((df['signal'] > 0) & df['extreme_momentum'])
            )
            df.loc[ultra_quality, 'signal_quality'] = 3
            
            # Report
            total_signals = (df['signal'] > 0).sum()
            bull_signals = df[~df['spy_bear']]['signal'].gt(0).sum() if 'spy_bear' in df else total_signals
            bear_signals = df[df['spy_bear']]['signal'].gt(0).sum() if 'spy_bear' in df else 0
            
            print(f"  Total signals: {total_signals}")
            print(f"  Bull market signals: {bull_signals}")
            print(f"  Bear market signals: {bear_signals}")
        
        return all_data
    
    def backtest(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        """Backtest with SIMPLE bear protection"""
        print(f"\n{'='*80}")
        print("RUNNING BACKTEST - Simple Bear Protection")
        print(f"{'='*80}\n")
        
        trades = []
        positions = {}
        capital = self.initial_capital
        
        all_dates = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        
        equity_curve = []
        monthly_trade_count = {}
        
        for current_date in all_dates:
            year_month = current_date.strftime('%Y-%m')
            if year_month not in monthly_trade_count:
                monthly_trade_count[year_month] = 0
            
            current_prices = {}
            is_bear = False
            
            for symbol, df in all_data.items():
                if current_date in df.index:
                    current_prices[symbol] = df.loc[current_date]
                    is_bear = df.loc[current_date].get('spy_bear', False)
            
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
                    
                    # Tighten on momentum
                    if profit_pct > 0.05 and row.get('extreme_momentum', False):
                        position['take_profit'] = current_price * 1.04
                
                # Exit conditions
                exit_trade = False
                exit_reason = ""
                
                if current_price <= position['stop_loss']:
                    exit_trade = True
                    exit_reason = "Stop Loss"
                elif current_price >= position['take_profit']:
                    exit_trade = True
                    exit_reason = "Take Profit"
                elif row.get('exit_signal', False):
                    exit_trade = True
                    exit_reason = "Exit Signal"
                elif days_held > 45:
                    exit_trade = True
                    exit_reason = "Time Exit"
                
                # SIMPLE bear protection: exit faster in bear
                if is_bear and days_held > self.bear_time_exit:
                    exit_trade = True
                    exit_reason = "Bear Time Exit"
                
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
                        'signal_quality': position['signal_quality'],
                        'was_bear': position['was_bear']
                    })
                    
                    monthly_trade_count[year_month] += 1
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del positions[symbol]
            
            # === SIMPLE max positions based on bear ===
            max_positions = self.bear_max_positions if is_bear else self.max_positions
            
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
            if len(positions) < max_positions:
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
                
                for opp in entry_opportunities[:max_positions - len(positions)]:
                    symbol = opp['symbol']
                    row = opp['row']
                    current_price = opp['price']
                    signal_type = opp['signal_type']
                    
                    # SIMPLE position sizing
                    base_risk = self.base_risk_pct
                    
                    # SIMPLE bear adjustment (50% reduction, not 70-90%)
                    if is_bear:
                        base_risk *= self.bear_risk_multiplier
                    
                    # Volatility adjustment (less aggressive than V3)
                    if row.get('high_volatility_stock', False):
                        base_risk *= self.high_vol_reduction
                    
                    # Quality adjustment
                    if opp['signal_quality'] == 3:
                        risk_pct = base_risk * self.momentum_multiplier
                    elif opp['signal_quality'] == 2:
                        risk_pct = base_risk * self.quality_multiplier
                    else:
                        risk_pct = base_risk
                    
                    # SIMPLE stop adjustment in bear
                    stop_pct = self.stop_loss_pct
                    if is_bear:
                        stop_pct *= self.bear_stop_multiplier  # 1.2% instead of 1.5%
                    
                    target_pct = self.take_profit_pct
                    
                    # Calculate position
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
                            'signal_quality': opp['signal_quality'],
                            'was_bear': is_bear
                        }
            
            # Track equity
            unrealized_pnl = sum(
                (current_prices[sym]['close'] - pos['entry_price']) * pos['shares'] * self.options_multiplier
                for sym, pos in positions.items()
                if sym in current_prices
            )
            current_equity = capital + unrealized_pnl
            equity_curve.append(current_equity)
        
        # Add to dataframe
        if all_data:
            first_symbol = list(all_data.keys())[0]
            all_data[first_symbol]['portfolio_equity'] = pd.Series(equity_curve, index=all_dates)
        
        self.monthly_trades = monthly_trade_count
        
        return trades, all_data
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Analysis"""
        print("\n" + "="*80)
        print("SIMPLIFIED MOMENTUM STRATEGY V4 - RESULTS")
        print("="*80)
        
        if not trades:
            print("\nNo trades executed.")
            return
        
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['year'] = trades_df['entry_date'].dt.year
        
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
        
        # Bear vs bull
        if 'was_bear' in trades_df.columns:
            print("\nPERFORMANCE BY MARKET CONDITION:")
            print("-"*70)
            print(f"{'Condition':<15} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'Avg Return%':<15}")
            print("-"*70)
            
            for condition in [False, True]:
                cond_name = "Bear Market" if condition else "Bull Market"
                cond_trades = trades_df[trades_df['was_bear'] == condition]
                if len(cond_trades) > 0:
                    cond_pnl = cond_trades['pnl'].sum()
                    cond_wr = len(cond_trades[cond_trades['pnl'] > 0]) / len(cond_trades) * 100
                    cond_avg_ret = cond_trades['return_pct'].mean()
                    
                    print(f"{cond_name:<15} {len(cond_trades):<10} {cond_wr:<10.1f} "
                          f"${cond_pnl:<15,.2f} {cond_avg_ret:<15.1f}")
        
        # By symbol
        print("\nPERFORMANCE BY SYMBOL:")
        print("-"*70)
        print(f"{'Symbol':<10} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'Avg Days':<10}")
        print("-"*70)
        
        for symbol in sorted(trades_df['symbol'].unique()):
            sym_trades = trades_df[trades_df['symbol'] == symbol]
            sym_pnl = sym_trades['pnl'].sum()
            sym_wr = len(sym_trades[sym_trades['pnl'] > 0]) / len(sym_trades) * 100
            sym_days = sym_trades['days_held'].mean()
            
            print(f"{symbol:<10} {len(sym_trades):<10} {sym_wr:<10.1f} ${sym_pnl:<15,.2f} {sym_days:<10.1f}")
        
        # Yearly
        print("\n" + "="*80)
        print("YEARLY PERFORMANCE:")
        print("="*80)
        print(f"{'Year':<10} {'Trades':<10} {'Win%':<10} {'P&L':<15} {'Return%':<10}")
        print("-"*70)
        
        cumulative_capital = self.initial_capital
        for year in sorted(trades_df['year'].unique()):
            year_trades = trades_df[trades_df['year'] == year]
            year_pnl = year_trades['pnl'].sum()
            year_return = (year_pnl / cumulative_capital) * 100
            year_wr = len(year_trades[year_trades['pnl'] > 0]) / len(year_trades) * 100
            
            year_marker = ""
            if year in [2021, 2022]:
                year_marker = " ← TARGET"
            
            print(f"{year:<10} {len(year_trades):<10} {year_wr:<10.1f} "
                  f"${year_pnl:<15,.2f} {year_return:<10.1f}{year_marker}")
            
            cumulative_capital += year_pnl
        
        print("\n" + "="*80)
        print("V4 SIMPLIFICATIONS:")
        print("  ✓ Pure momentum only (NO mean reversion)")
        print("  ✓ Simple bear detection (SPY < 200 MA)")
        print("  ✓ 50% position reduction in bear (not 70-90%)")
        print("  ✓ 3 max positions in bear (not 1-2)")
        print("  ✓ Keeps V2's proven momentum strategies")
        print("="*80)

def run_simplified_strategy():
    """Run simplified momentum-only strategy"""
    print("\n" + "="*80)
    print("SIMPLIFIED MOMENTUM STRATEGY V4")
    print("Pure momentum + Simple bear protection")
    print("="*80)
    
    strategy = SimplifiedMomentumStrategy(
        # symbols=["NVDA", "ORCL", "TSLA", "PLTR", "IBM"],
        # symbols=["NVDA", "TSLA", "PLTR", "AMD", "COIN"],
        symbols=["NVDA",  "PLTR", "AMD", "TSLA"],    
        start_date="2020-01-01",
        initial_capital=10000,
        use_bear_protection=True
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
    trades, data = run_simplified_strategy()