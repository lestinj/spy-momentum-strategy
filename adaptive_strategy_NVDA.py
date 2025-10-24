"""
Fixed Aggressive NVDA Strategy - Simplified for Execution
Targeting aggressive returns with focus on 2025 sideways market
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class AggressiveNVDAStrategy:
    def __init__(
        self,
        symbol: str = "NVDA",
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 10000,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # AGGRESSIVE PARAMETERS
        self.fast_ma = 5   # Very fast
        self.slow_ma = 20  # Medium
        self.trend_ma = 50
        
        # Tight stops, bigger targets
        self.stop_loss_pct = 0.01  # 1% stop
        self.take_profit_pct = 0.04  # 4% target (4:1 R:R)
        
        # Aggressive sizing
        self.base_risk_pct = 0.05  # 5% risk
        self.max_risk_pct = 0.10   # 10% max
        
        # Mean reversion
        self.rsi_oversold = 40
        self.rsi_overbought = 60
        
        self.options_multiplier = 100
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        print(f"\nLoading {self.symbol} data...")
        
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        extended_start = start - pd.Timedelta(days=100)
        
        # Get NVDA data
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=extended_start, end=end, interval='1d')
        
        # Clean columns
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Simple indicators
        df['sma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
        df['sma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
        df['sma_trend'] = df['close'].rolling(window=self.trend_ma).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (tighter for mean reversion)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (1.5 * df['bb_std'])  # 1.5 std
        df['bb_lower'] = df['bb_middle'] - (1.5 * df['bb_std'])
        
        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['high_volume'] = df['volume'] > df['volume_sma'] * 1.2
        
        # Market regime - simple approach
        df['range_50'] = df['high'].rolling(50).max() - df['low'].rolling(50).min()
        df['range_pct'] = df['range_50'] / df['close']
        df['sideways'] = df['range_pct'] < 0.15  # Less than 15% range
        
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # Filter to trading period
        df = df[start:end]
        
        print(f"✓ Loaded {len(df)} trading days")
        
        # Debug info
        print(f"\nData check:")
        print(f"  Sideways days: {df['sideways'].sum()} ({df['sideways'].sum()/len(df)*100:.1f}%)")
        print(f"  RSI range: {df['rsi'].min():.1f} - {df['rsi'].max():.1f}")
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate aggressive signals"""
        
        # Initialize
        df['signal'] = 0
        df['signal_type'] = 'none'
        
        # === MOMENTUM TRADES (Trending) ===
        momentum_long = (
            (df['sma_fast'] > df['sma_slow']) &
            (df['close'] > df['sma_trend']) &
            (df['momentum_5'] > 0.02) &  # 2% in 5 days
            (df['rsi'] > 50) & (df['rsi'] < 70)
        )
        
        # === MEAN REVERSION (Key for sideways) ===
        # Long when oversold
        reversion_long = (
            (df['close'] < df['bb_lower']) |  # Below lower band OR
            ((df['rsi'] < self.rsi_oversold) & (df['close'] < df['sma_slow']))  # Oversold
        ) & (df['close'] > df['sma_trend'] * 0.95)  # Not too far from trend
        
        # Short when overbought (for ranging markets)
        reversion_short = (
            (df['close'] > df['bb_upper']) |  # Above upper band OR
            ((df['rsi'] > self.rsi_overbought) & (df['close'] > df['sma_slow']))
        ) & (df['close'] < df['sma_trend'] * 1.05) & df['sideways']  # Only in sideways
        
        # === BREAKOUT TRADES ===
        breakout_long = (
            (df['close'] > df['high'].shift(1)) &
            (df['momentum_5'] > 0.03) &
            (df['high_volume']) &
            (df['sma_fast'] > df['sma_slow'])
        )
        
        # === EXTREME OVERSOLD BOUNCE ===
        bounce_long = (
            (df['rsi'] < 30) &
            (df['momentum_5'] < -0.05) &  # Down 5% in 5 days
            (df['close'] > df['sma_trend'] * 0.9)  # Still relatively near trend
        )
        
        # Assign signals
        df.loc[momentum_long, 'signal'] = 1
        df.loc[momentum_long, 'signal_type'] = 'momentum'
        
        df.loc[reversion_long, 'signal'] = 1  
        df.loc[reversion_long, 'signal_type'] = 'reversion'
        
        df.loc[reversion_short, 'signal'] = -1
        df.loc[reversion_short, 'signal_type'] = 'reversion_short'
        
        df.loc[breakout_long, 'signal'] = 1
        df.loc[breakout_long, 'signal_type'] = 'breakout'
        
        df.loc[bounce_long, 'signal'] = 1
        df.loc[bounce_long, 'signal_type'] = 'bounce'
        
        # Signal analysis
        print(f"\nSignal Generation:")
        print(f"  Momentum: {(df['signal_type'] == 'momentum').sum()}")
        print(f"  Reversion Long: {((df['signal_type'] == 'reversion') & (df['signal'] == 1)).sum()}")
        print(f"  Reversion Short: {(df['signal_type'] == 'reversion_short').sum()}")
        print(f"  Breakout: {(df['signal_type'] == 'breakout').sum()}")
        print(f"  Bounce: {(df['signal_type'] == 'bounce').sum()}")
        print(f"  Total Signals: {(df['signal'] != 0).sum()}")
        
        return df
    
    def backtest(self, df: pd.DataFrame) -> Tuple[List[Dict], pd.DataFrame]:
        """Run aggressive backtest"""
        trades = []
        position = None
        capital = self.initial_capital
        equity_curve = []
        
        for idx, row in df.iterrows():
            current_price = row['close']
            
            # Manage position
            if position is not None:
                days_held = (idx - position['entry_date']).days
                
                if position['direction'] == 1:  # Long
                    pnl = (current_price - position['entry_price']) * position['shares'] * self.options_multiplier
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:  # Short
                    pnl = (position['entry_price'] - current_price) * position['shares'] * self.options_multiplier
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
                # Exit logic
                should_exit = False
                exit_reason = ""
                
                # Stop loss
                if position['direction'] == 1 and current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif position['direction'] == -1 and current_price >= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop Loss"
                
                # Take profit
                elif position['direction'] == 1 and current_price >= position['take_profit']:
                    should_exit = True
                    exit_reason = "Take Profit"
                elif position['direction'] == -1 and current_price <= position['take_profit']:
                    should_exit = True
                    exit_reason = "Take Profit"
                
                # Quick exit for mean reversion
                elif position['strategy'] in ['reversion', 'reversion_short'] and pnl_pct > 0.015:
                    should_exit = True
                    exit_reason = "Quick Profit"
                
                # Time exit
                elif days_held > 10:
                    should_exit = True
                    exit_reason = "Time Exit"
                
                # Trend change exit
                elif position['direction'] == 1 and row['sma_fast'] < row['sma_slow']:
                    should_exit = True
                    exit_reason = "Trend Change"
                
                if should_exit:
                    capital += pnl
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': idx,
                        'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                        'strategy': position['strategy'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return_pct': pnl_pct * 100,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'year': idx.year
                    })
                    
                    position = None
            
            # New entry
            if position is None and row['signal'] != 0 and capital > 1000:
                # Position sizing
                risk_pct = self.base_risk_pct
                
                # Double size for high conviction
                if row['signal_type'] in ['bounce', 'reversion'] and row['sideways']:
                    risk_pct = min(risk_pct * 1.5, self.max_risk_pct)
                
                # Calculate position
                if row['signal'] == 1:  # Long
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                    direction = 1
                else:  # Short
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
                    direction = -1
                
                risk_per_share = abs(current_price - stop_loss) * self.options_multiplier
                risk_amount = capital * risk_pct
                shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
                
                # Max position 70% of capital
                max_value = capital * 0.7
                max_shares = max_value / (current_price * self.options_multiplier)
                shares = min(shares, max_shares)
                
                if shares > 0.1:
                    position = {
                        'entry_date': idx,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'shares': shares,
                        'direction': direction,
                        'strategy': row['signal_type']
                    }
            
            # Track equity
            if position:
                if position['direction'] == 1:
                    unrealized = (current_price - position['entry_price']) * position['shares'] * self.options_multiplier
                else:
                    unrealized = (position['entry_price'] - current_price) * position['shares'] * self.options_multiplier
                current_equity = capital + unrealized
            else:
                current_equity = capital
                
            equity_curve.append(current_equity)
        
        df['equity'] = equity_curve
        
        return trades, df
    
    def analyze_results(self, trades: List[Dict], df: pd.DataFrame):
        """Analyze results"""
        print("\n" + "="*80)
        print("AGGRESSIVE STRATEGY RESULTS - TARGETING HIGH RETURNS")
        print("="*80)
        
        if not trades:
            print("\nNo trades executed - check signal generation")
            return
        
        trades_df = pd.DataFrame(trades)
        
        # Overall metrics
        final_equity = df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = ((final_equity / self.initial_capital) ** (1/years) - 1) * 100
        
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Initial Capital: ${self.initial_capital:,.0f}")
        print(f"  Final Equity: ${final_equity:,.0f}")
        print(f"  Total Return: {total_return:.1f}%")
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        
        # Strategy breakdown
        print(f"\nSTRATEGY BREAKDOWN:")
        for strategy in trades_df['strategy'].unique():
            strat_trades = trades_df[trades_df['strategy'] == strategy]
            strat_pnl = strat_trades['pnl'].sum()
            strat_wr = len(strat_trades[strat_trades['pnl'] > 0]) / len(strat_trades) * 100
            print(f"  {strategy}: {len(strat_trades)} trades, {strat_wr:.1f}% WR, ${strat_pnl:,.0f}")
        
        # Yearly breakdown
        print(f"\nYEARLY PERFORMANCE:")
        for year in sorted(trades_df['year'].unique()):
            year_trades = trades_df[trades_df['year'] == year]
            year_pnl = year_trades['pnl'].sum()
            year_wr = len(year_trades[year_trades['pnl'] > 0]) / len(year_trades) * 100
            print(f"  {year}: {len(year_trades)} trades, {year_wr:.1f}% WR, ${year_pnl:,.0f}")
        
        # Direction analysis
        longs = trades_df[trades_df['direction'] == 'LONG']
        shorts = trades_df[trades_df['direction'] == 'SHORT']
        
        print(f"\nDIRECTION ANALYSIS:")
        print(f"  Longs: {len(longs)} trades, ${longs['pnl'].sum():,.0f} P&L")
        print(f"  Shorts: {len(shorts)} trades, ${shorts['pnl'].sum():,.0f} P&L")
        
        print(f"\nTARGET STATUS:")
        print(f"  300% Annual Target: {'✅ ACHIEVED!' if cagr >= 300 else '❌ MISSED'}")
        print(f"  100% in 6 Months: {'✅ POSSIBLE!' if cagr >= 200 else '🎯 WORKING ON IT'}")
        
        print("\n" + "="*80)

def run_aggressive_strategy():
    strategy = AggressiveNVDAStrategy(
        symbol="NVDA",
        start_date="2022-01-01",
        initial_capital=10000
    )
    
    df = strategy.load_data()
    df = strategy.generate_signals(df)
    trades, df = strategy.backtest(df)
    strategy.analyze_results(trades, df)
    
    return trades, df

if __name__ == "__main__":
    trades, df = run_aggressive_strategy()