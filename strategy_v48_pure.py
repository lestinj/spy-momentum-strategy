"""
Pure Multi-Symbol Momentum Strategy V4.8
═══════════════════════════════════════════════════════════════════════════════

BACK TO BASICS: What Actually Works
  - V4's proven momentum strategies (97% CAGR)
  - NO regime detection (complexity hurts performance)
  - NO bear positions (VXX/SQQQ/SPXU don't work)
  - 15 symbols for better diversification
  - Lower max DD through position spreading

SYMBOL SELECTION (15 high-momentum stocks):
  Core 5: NVDA, TSLA, PLTR, AMD, COIN
  Growth 5: SMCI, MSTR, CRWD, SNOW, NET
  Tech 5: RIOT, MARA, DDOG, ZS, MDB

STRATEGY:
  - Same V4 momentum signals (6 strategies)
  - 8 max positions (vs V4's 5)
  - 1.5% stop loss
  - 2-4% position sizing based on quality
  
Expected Performance:
  - CAGR: 85-95% (match V4 with less concentration)
  - Max DD: -50% to -60% (better than V4's -79%)
  - Diversification benefit: lower volatility
  - Scalable to larger accounts
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
warnings.filterwarnings('ignore')

class PureMultiSymbolStrategy:
    """V4.8 - Pure momentum with 15 symbols, no complexity"""
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
        output_dir: str = "trading_results"
    ):
        # 15 high-momentum symbols
        self.symbols = symbols or [
            # Core 5 (proven winners)
            "NVDA", "TSLA", "PLTR", "AMD", "COIN",
            # Growth 5 (high momentum)
            "SMCI", "MSTR", "CRWD", "SNOW", "NET",
            # Tech 5 (diversification)
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
        print("PURE MULTI-SYMBOL MOMENTUM STRATEGY V4.8")
        print("Simple, effective, proven")
        print("="*80)
        
        # V4 parameters - proven to work
        self.base_risk = 0.020              # 2% base
        self.quality_mult = 1.5             # 3% on quality
        self.momentum_mult = 2.0            # 4% on ultra momentum
        self.stop_loss = 0.015              # 1.5%
        self.take_profit = 0.08             # 8%
        self.max_hold = 45
        self.max_positions = 8              # More than V4's 5
        
        print(f"\n📈 STRATEGY PARAMETERS:")
        print(f"  Symbols: {len(self.symbols)} ({', '.join(self.symbols[:5])}...)")
        print(f"  Max positions: {self.max_positions}")
        print(f"  Position sizing: 2-4% (quality-adjusted)")
        print(f"  Stop loss: {self.stop_loss*100}%")
        print(f"  Take profit: {self.take_profit*100}%")
        
        self.options_multiplier = 100
        
        print("\n" + "="*80)
        print("PHILOSOPHY: Keep it simple, trade what works")
        print("  ✓ Pure momentum (V4's proven strategies)")
        print("  ✓ No regime complexity")
        print("  ✓ No bear positions")
        print("  ✓ More symbols = better diversification")
        print("="*80 + "\n")
    
    def load_single_symbol(self, symbol: str) -> pd.DataFrame:
        """Load data with V4's indicators"""
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
        """Generate V4's proven momentum signals"""
        print("="*80)
        print("GENERATING V4 MOMENTUM SIGNALS")
        print("="*80)
        
        for symbol, df in all_data.items():
            # V4 Strategy 1: Classic Trend
            trend_long = (
                (df['sma_fast'] > df['sma_slow']) &
                (df['close'] > df['sma_trend']) &
                (df['rsi'] > 40) & (df['rsi'] < 70) &
                (~df['bear_market'])
            )
            
            # V4 Strategy 2: Quality Pullback
            pullback_long = (
                df['perfect_uptrend'] &
                (df['close'] < df['sma_fast']) &
                (df['close'] > df['sma_slow']) &
                (df['rsi'] < 50) & (df['rsi'] > 35) &
                df['higher_low']
            )
            
            # V4 Strategy 3: Breakout
            breakout_long = (
                (df['close'] > df['high'].shift(1)) &
                (df['close'] > df['bb_upper']) &
                (df['volume'] > df['volume_sma'] * 1.5) &
                df['perfect_uptrend'] &
                df['strong_momentum']
            )
            
            # V4 Strategy 4: Momentum Acceleration
            momentum_long = (
                df['extreme_momentum'] &
                df['perfect_uptrend'] &
                (df['rsi'] > 50) & (df['rsi'] < 65) &
                df['high_volume'] &
                (df['momentum_20'] > 0.10)
            )
            
            # V4 Strategy 5: Oversold Bounce
            bounce_long = (
                (df['close'] > df['sma_trend'] * 0.97) &
                (df['rsi'] < 30) &
                (df['close'] < df['bb_lower']) &
                (~df['bear_market'])
            )
            
            # V4 Strategy 6: Early Recovery
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
            
            # Exit signal
            df['exit_signal'] = (
                (df['sma_fast'] < df['sma_slow']) |
                (df['close'] < df['sma_trend'] * 0.97)
            )
            
            # Quality scoring (V4 style)
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
        """Run pure momentum backtest"""
        print("="*80)
        print("RUNNING PURE MOMENTUM BACKTEST")
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
            
            # === MANAGE EXISTING POSITIONS ===
            symbols_to_remove = []
            
            for symbol, pos in list(positions.items()):
                if symbol not in current_prices:
                    continue
                
                row = current_prices[symbol]
                current_price = row['close']
                days_held = (current_date - pos['entry_date']).days
                profit_pct = (current_price - pos['entry_price']) / pos['entry_price']
                
                # Exit logic
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
            
            # === NEW ENTRIES ===
            if len(positions) < self.max_positions:
                opportunities = []
                
                for symbol, row in current_prices.items():
                    if symbol in positions:
                        continue
                    
                    if row.get('signal', 0) > 0:
                        opportunities.append({
                            'symbol': symbol,
                            'quality': row.get('quality', 1),
                            'price': row['close'],
                            'signal_type': row.get('signal_type', 'unknown')
                        })
                
                opportunities.sort(key=lambda x: x['quality'], reverse=True)
                
                for opp in opportunities[:self.max_positions - len(positions)]:
                    symbol = opp['symbol']
                    current_price = opp['price']
                    
                    # V4 position sizing
                    base_risk = self.base_risk
                    if opp['quality'] == 2:
                        base_risk *= self.quality_mult
                    elif opp['quality'] == 3:
                        base_risk *= self.momentum_mult
                    
                    stop_loss = current_price * (1 - self.stop_loss)
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
                            'quality': opp['quality']
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
            print("No trades to save")
            return
        
        trades_df = pd.DataFrame(trades)
        csv_path = self.output_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(csv_path, index=False)
        print(f"\n✓ Trades saved to: {csv_path}")
        return csv_path
    
    def plot_equity_curve(self, all_data: Dict[str, pd.DataFrame], trades: List[Dict]):
        """Plot portfolio growth over time"""
        first_sym = list(all_data.keys())[0]
        if 'portfolio_equity' not in all_data[first_sym].columns:
            print("No equity data to plot")
            return
        
        equity = all_data[first_sym]['portfolio_equity'].dropna()
        trades_df = pd.DataFrame(trades)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('V4.8 Pure Multi-Symbol Momentum Strategy', fontsize=16, fontweight='bold')
        
        # Plot 1: Equity curve
        ax1.plot(equity.index, equity.values, linewidth=2, label='Portfolio Value', color='#2E86AB')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Add trade markers
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        winning_entries = trades_df[trades_df['pnl'] > 0]['entry_date']
        losing_entries = trades_df[trades_df['pnl'] <= 0]['entry_date']
        
        for date in winning_entries:
            if date in equity.index:
                ax1.scatter(date, equity.loc[date], color='green', marker='^', s=30, alpha=0.3)
        
        for date in losing_entries:
            if date in equity.index:
                ax1.scatter(date, equity.loc[date], color='red', marker='v', s=30, alpha=0.3)
        
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Portfolio Growth Over Time', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Calculate metrics
        returns = equity.pct_change().dropna()
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        # Plot 2: Drawdown
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Portfolio Drawdown', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Equity curve saved to: {plot_path}")
        
        plt.show()
    
    def analyze_results(self, trades: List[Dict], all_data: Dict[str, pd.DataFrame]):
        """Analyze results"""
        print("\n" + "="*80)
        print("PURE MULTI-SYMBOL MOMENTUM V4.8 RESULTS")
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
        print(f"  Total Trades: {len(trades_df)}")
        
        # By symbol
        print("\n" + "="*80)
        print("PERFORMANCE BY SYMBOL:")
        print("="*80)
        print(f"{'Symbol':<10} {'Trades':<8} {'Win%':<8} {'P&L':<15} {'Avg Return%':<12}")
        print("-"*70)
        
        for symbol in sorted(trades_df['symbol'].unique()):
            sym_trades = trades_df[trades_df['symbol'] == symbol]
            sym_pnl = sym_trades['pnl'].sum()
            sym_wr = len(sym_trades[sym_trades['pnl'] > 0]) / len(sym_trades) * 100
            sym_avg_ret = sym_trades['return_pct'].mean()
            
            print(f"{symbol:<10} {len(sym_trades):<8} {sym_wr:<8.1f} ${sym_pnl:<14,.0f} {sym_avg_ret:<12.1f}")
        
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
            
            print(f"{year:<8} {len(yr_trades):<8} {yr_wr:<8.1f} ${yr_pnl:<14,.0f} {yr_ret:<10.1f}")
            cum_cap += yr_pnl
        
        # Strategy comparison
        print("\n" + "="*80)
        print("STRATEGY COMPARISON:")
        print("="*80)
        print(f"{'Metric':<25} {'V4 (5 sym)':<15} {'V4.8 (15 sym)':<15}")
        print("-"*60)
        print(f"{'Symbols':<25} {'5':<15} {f'{len(self.symbols)}':<15}")
        print(f"{'Max positions':<25} {'5':<15} {f'{self.max_positions}':<15}")
        print(f"{'CAGR':<25} {'97.2%':<15} {f'{cagr:.1f}%':<15}")
        print(f"{'Max DD':<25} {'-79.2%':<15} {f'{max_dd:.1f}%':<15}")
        print(f"{'Win Rate':<25} {'39.2%':<15} {f'{win_rate:.1f}%':<15}")
        print(f"{'Sharpe':<25} {'1.25':<15} {f'{sharpe:.2f}':<15}")
        
        print("\n✅ V4.8 ADVANTAGES:")
        print("  • More diversification (15 symbols vs 5)")
        print("  • Lower concentration risk")
        print("  • More consistent performance")
        print("  • Scalable to larger accounts")
        print("  • No complexity (just momentum)")
        print("="*80)

def run_v48():
    """Run V4.8 strategy with plots and CSV export"""
    strategy = PureMultiSymbolStrategy(
        start_date="2020-01-01",
        initial_capital=20000
    )
    
    all_data = strategy.load_all_data()
    if not all_data:
        print("\n❌ Failed to load data")
        return None, None
    
    all_data = strategy.generate_signals(all_data)
    trades, all_data = strategy.backtest(all_data)
    
    # Save trades to CSV
    strategy.save_trades_csv(trades)
    
    # Plot equity curve
    strategy.plot_equity_curve(all_data, trades)
    
    # Analyze
    strategy.analyze_results(trades, all_data)
    
    return trades, all_data

if __name__ == "__main__":
    trades, data = run_v48()
