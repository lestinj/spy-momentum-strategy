#!/usr/bin/env python3
"""
ML-ENHANCED V49 - OPTIMAL VERSION
==================================
Uses ML for intelligent position SIZING, not filtering
Takes ALL V49 signals but sizes them based on ML confidence
Target: Beat 108% CAGR consistently

KEY STRATEGY:
- Never skip V49 signals (they're proven profitable)
- Use ML confidence to modulate position size (10-50%)
- Graduated responses, not binary decisions
- Risk management through sizing, not elimination
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Unified configuration support
import os
try:
    from trading_config import TradingConfig
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    UNIFIED_CONFIG_AVAILABLE = False


class MLOptimizedV49:
    """
    Optimal ML enhancement that SIZES positions, doesn't FILTER trades
    """
    
    def __init__(self, initial_capital=None, config=None):
        """
        Initialize with either unified config or traditional capital parameter
        
        Priority:
        1. config parameter (TradingConfig object)
        2. initial_capital parameter
        3. Default to 30000
        """
        # Load from unified config if provided
        if config is not None:
            self.config = config
            self.initial_capital = config.total_capital
            self.symbols = config.symbols
            self.max_positions = config.max_positions
            self.leverage = config.max_leverage
            self.base_stop_loss = config.base_stop_loss
            self.base_take_profit = config.base_take_profit
            self.rsi_period = config.rsi_period
            self.rsi_buy = config.rsi_buy
            self.rsi_sell = config.rsi_sell
            self.ma_fast = config.ma_fast
            self.ma_slow = config.ma_slow
            self.high_confidence = config.high_confidence
            self.low_confidence = config.low_confidence
            
            print(f"📊 Using unified config:")
            print(f"   Capital: ${self.initial_capital:,.0f}")
            print(f"   Leverage: {self.leverage}x")
            print(f"   Max Positions: {self.max_positions}")
        else:
            # Use provided capital or default
            self.config = None
            self.initial_capital = initial_capital if initial_capital is not None else 30000
            
            # Default parameters (matching original)
            self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
            self.rsi_period = 14
            self.rsi_buy = 55
            self.rsi_sell = 45
            self.ma_fast = 10
            self.ma_slow = 30
            self.max_positions = 3
            self.leverage = 2.0
            self.base_stop_loss = 0.08
            self.base_take_profit = 0.25
            self.high_confidence = 0.60
            self.low_confidence = 0.30
        
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # BASE position sizing (will be adjusted by ML)
        self.base_position_size = 0.30  # 30% base
        
        # DYNAMIC SIZING RANGE based on ML confidence
        # Key insight: ALWAYS take position, just vary the size
        self.min_position_size = 0.10   # Even low confidence gets 10%
        self.max_position_size = 0.50   # High confidence gets 50%
        
        # ML model settings
        self.min_ml_accuracy = 0.45    # Use ML even with moderate accuracy
        self.stock_accuracies = {}
        self.models = {}
        self.scalers = {}
        self.features = {}
        self.data = {}
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self, df):
        """Comprehensive feature engineering for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Core momentum features
        features['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        features['rsi_change'] = features['rsi'].diff()
        features['rsi_ma'] = features['rsi'].rolling(5).mean()
        
        # Moving averages
        features['ma_fast'] = df['Close'].rolling(self.ma_fast).mean()
        features['ma_slow'] = df['Close'].rolling(self.ma_slow).mean()
        features['ma_ratio'] = features['ma_fast'] / features['ma_slow']
        features['price_to_ma_fast'] = df['Close'] / features['ma_fast']
        features['price_to_ma_slow'] = df['Close'] / features['ma_slow']
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_price'] = features['volume_ratio'] * df['Close'].pct_change()
        features['volume_surge'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
        
        # Trend strength at multiple timeframes
        for period in [5, 10, 20, 50]:
            features[f'trend_strength_{period}'] = (df['Close'] - df['Close'].rolling(period).min()) / \
                                         (df['Close'].rolling(period).max() - df['Close'].rolling(period).min() + 0.0001)
        
        # Momentum at multiple periods
        for period in [3, 5, 10, 20]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Volatility features
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_regime'] = features['volatility'] / features['volatility'].rolling(60).mean()
        
        # Market microstructure
        features['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
        features['close_to_high'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 0.0001)
        features['range_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                                     (df['High'].rolling(20).max() - df['Low'].rolling(20).min() + 0.0001)
        
        # Time-based features
        features['day_of_week'] = pd.to_datetime(df.index).dayofweek
        features['month'] = pd.to_datetime(df.index).month
        
        return features.fillna(0)
    
    def train_ml_models(self, symbol, df, features_df):
        """Train ML model for confidence scoring"""
        print(f"Training ML for {symbol}...")
        
        # Create labels - will it go up 3% in next 5 days?
        future_returns = df['Close'].shift(-5) / df['Close'] - 1
        labels = (future_returns > 0.03).astype(int)
        
        # Prepare data
        X = features_df.fillna(0)
        y = labels.fillna(0)
        
        # Remove invalid rows
        valid_mask = ~(y.isna()) & (X.index >= df.index[100])
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 150:
            self.stock_accuracies[symbol] = 0.0
            return False
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Store model
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Test accuracy
        accuracy = model.score(X_test_scaled, y_test)
        self.stock_accuracies[symbol] = accuracy
        
        status = "✓" if accuracy >= self.min_ml_accuracy else "○"
        print(f"  {symbol} ML Accuracy: {accuracy:.2%} {status}")
        
        return True
    
    def get_v49_signals(self, date):
        """Get original V49 signals - these are proven profitable"""
        signals = []
        
        for symbol, df in self.data.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 50:  # Need history
                continue
            
            try:
                current = df.loc[date]
                
                # Extract values
                rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                close = float(current['Close'])
                ma_fast = float(current['MA_Fast']) if not pd.isna(current['MA_Fast']) else 0
                ma_slow = float(current['MA_Slow']) if not pd.isna(current['MA_Slow']) else 0
                
                if rsi == 0 or ma_fast == 0 or ma_slow == 0:
                    continue
                
                signal_type = None
                
                # STRATEGY 1: Trend Following
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    signal_type = 'TREND_FOLLOW'
                
                # STRATEGY 2: Pullback Entry
                elif (rsi < self.rsi_buy and rsi > 45 and
                      close > ma_slow and
                      ma_fast > ma_slow):
                    signal_type = 'PULLBACK'
                
                if signal_type:
                    signals.append({
                        'symbol': symbol,
                        'price': close,
                        'date': date,
                        'strategy': signal_type,
                        'rsi': rsi
                    })
                    
            except Exception:
                continue
        
        return signals
    
    def get_ml_confidence(self, symbol, date):
        """Get ML confidence score for position sizing"""
        # Default confidence if no ML available
        default_confidence = 0.5
        
        # Check if ML should be used for this stock
        if symbol not in self.stock_accuracies:
            return default_confidence
        
        if self.stock_accuracies[symbol] < self.min_ml_accuracy:
            return default_confidence
        
        if symbol not in self.models or symbol not in self.features:
            return default_confidence
        
        try:
            features_df = self.features[symbol]
            if date not in features_df.index:
                return default_confidence
            
            X = features_df.loc[date].values.reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            
            # Get prediction probability
            ml_prob = self.models[symbol].predict_proba(X_scaled)[0]
            confidence = ml_prob[1]  # Probability of positive return
            
            return confidence
            
        except Exception:
            return default_confidence
    
    def calculate_dynamic_position_size(self, ml_confidence):
        """
        Calculate position size based on ML confidence
        KEY: Always return some position, never zero
        """
        if ml_confidence >= self.high_confidence:
            # High confidence: Maximum position
            position_size = self.max_position_size
        elif ml_confidence <= self.low_confidence:
            # Low confidence: Minimum position (but still take it!)
            position_size = self.min_position_size
        else:
            # Linear interpolation between min and max
            confidence_range = self.high_confidence - self.low_confidence
            confidence_normalized = (ml_confidence - self.low_confidence) / confidence_range
            size_range = self.max_position_size - self.min_position_size
            position_size = self.min_position_size + (size_range * confidence_normalized)
        
        return position_size
    
    def calculate_dynamic_stops(self, ml_confidence, position_size):
        """
        Adjust stop loss and take profit based on confidence and position size
        Smaller positions get tighter stops, larger positions get more room
        """
        # Inverse relationship: smaller position = tighter stop
        if position_size >= 0.40:
            # Large position: Give it room to work
            stop_loss = 0.10
            take_profit = 0.30
        elif position_size >= 0.25:
            # Medium position: Standard stops
            stop_loss = 0.08
            take_profit = 0.25
        else:
            # Small position: Tight risk management
            stop_loss = 0.05
            take_profit = 0.15
        
        return stop_loss, take_profit
    
    def enhance_signals_with_ml(self, signals, date):
        """
        CRITICAL: Enhance ALL signals with ML-based sizing
        Never filter out signals, only adjust position size
        """
        enhanced_signals = []
        
        for signal in signals:
            symbol = signal['symbol']
            
            # Get ML confidence
            ml_confidence = self.get_ml_confidence(symbol, date)
            
            # ALWAYS enhance and include the signal
            signal['ml_confidence'] = ml_confidence
            
            # Calculate dynamic position size (NEVER ZERO)
            position_size = self.calculate_dynamic_position_size(ml_confidence)
            signal['position_size'] = position_size
            
            # Calculate dynamic stops based on position size
            stop_loss, take_profit = self.calculate_dynamic_stops(ml_confidence, position_size)
            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit
            
            # Add confidence level for tracking
            if ml_confidence >= self.high_confidence:
                signal['confidence_level'] = 'HIGH'
                signal['strategy'] += '_HIGH_CONF'
            elif ml_confidence <= self.low_confidence:
                signal['confidence_level'] = 'LOW'
                signal['strategy'] += '_LOW_CONF'
            else:
                signal['confidence_level'] = 'MEDIUM'
                signal['strategy'] += '_MED_CONF'
            
            # Calculate quality score for ranking
            base_quality = 3 if 'TREND_FOLLOW' in signal['strategy'] else 2
            signal['quality_score'] = base_quality * (0.5 + ml_confidence)
            
            # ALWAYS append the signal
            enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def check_exits(self, date):
        """Check exit conditions with dynamic parameters"""
        exits = []
        
        for symbol, position in list(self.positions.items()):
            if date not in self.data[symbol].index:
                continue
            
            try:
                current = self.data[symbol].loc[date]
                current_price = float(current['Close'])
                current_rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                days_held = (date - position['entry_date']).days
                
                # Use position-specific stops
                stop_loss = position.get('stop_loss', self.base_stop_loss)
                take_profit = position.get('take_profit', self.base_take_profit)
                
                exit_reason = None
                
                # Check exits
                if pnl_pct <= -stop_loss:
                    exit_reason = f'STOP_LOSS_{stop_loss*100:.0f}%'
                elif pnl_pct >= take_profit:
                    exit_reason = f'TAKE_PROFIT_{take_profit*100:.0f}%'
                elif current_rsi > 0 and current_rsi < self.rsi_sell:
                    exit_reason = 'RSI_SELL'
                elif days_held >= 14:
                    exit_reason = 'TIME_EXIT'
                
                if exit_reason:
                    exits.append({
                        'symbol': symbol,
                        'exit_price': current_price,
                        'date': date,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct
                    })
                    
            except Exception:
                continue
        
        return exits
    
    def execute_trade(self, signal):
        """Execute trade with dynamic position sizing"""
        # Use ML-determined position size
        position_size = signal.get('position_size', self.base_position_size)
        position_value = self.capital * position_size
        leveraged_value = position_value * self.leverage
        shares = int(leveraged_value / signal['price'])
        
        if shares > 0 and self.capital >= position_value:
            self.positions[signal['symbol']] = {
                'shares': shares,
                'entry_price': signal['price'],
                'entry_date': signal['date'],
                'entry_value': position_value,
                'leveraged_value': leveraged_value,
                'ml_confidence': signal.get('ml_confidence', 0.5),
                'position_size': position_size,
                'stop_loss': signal.get('stop_loss', self.base_stop_loss),
                'take_profit': signal.get('take_profit', self.base_take_profit),
                'strategy': signal['strategy']
            }
            
            self.capital -= position_value
    
    def execute_exit(self, exit_info):
        """Execute exit and record trade"""
        symbol = exit_info['symbol']
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_value = position['shares'] * exit_info['exit_price']
            
            # Calculate P&L
            pnl = exit_value - position['leveraged_value']
            actual_return = position['entry_value'] + pnl
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': exit_info['date'],
                'entry_price': position['entry_price'],
                'exit_price': exit_info['exit_price'],
                'shares': position['shares'],
                'pnl': pnl,
                'pnl_pct': exit_info['pnl_pct'],
                'strategy': position['strategy'],
                'ml_confidence': position['ml_confidence'],
                'position_size': position['position_size'],
                'exit_reason': exit_info['exit_reason']
            })
            
            # Return capital
            self.capital += actual_return
            
            # Remove position
            del self.positions[symbol]
    
    def load_data(self, start_date='2015-01-01', end_date=None):
        """Load and prepare data for all symbols"""
        print(f"\n📊 Loading ML-Optimized V49 Strategy...")
        print(f"Period: {start_date} to {end_date or 'present'}")
        
        self.data = {}
        self.features = {}
        
        for symbol in self.symbols:
            try:
                # Download with extended history for ML training
                extended_start = pd.to_datetime(start_date) - timedelta(days=200)
                df = yf.download(symbol, start=extended_start, end=end_date,
                               progress=False, auto_adjust=True)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) > 100:
                    # Add indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    # Prepare features
                    features = self.prepare_features(df)
                    
                    # Train ML on historical data
                    backtest_start = pd.to_datetime(start_date)
                    train_df = df[df.index < backtest_start]
                    train_features = features[features.index < backtest_start]
                    
                    if len(train_df) > 100:
                        self.train_ml_models(symbol, train_df, train_features)
                    
                    # Store data for backtest
                    self.data[symbol] = df[df.index >= backtest_start]
                    self.features[symbol] = features[features.index >= backtest_start]
                    
                    print(f"✓ {symbol}: Ready")
                    
            except Exception as e:
                print(f"✗ {symbol}: {e}")
        
        print(f"\n✓ Loaded {len(self.data)} symbols")
        
        # Show ML status
        if self.stock_accuracies:
            enabled = sum(1 for acc in self.stock_accuracies.values() if acc >= self.min_ml_accuracy)
            print(f"📊 ML Models: {enabled}/{len(self.stock_accuracies)} enabled (>{self.min_ml_accuracy*100:.0f}% accuracy)")
        
        return len(self.data) > 0
    
    def run_backtest(self):
        """Run the optimized backtest"""
        print("\n" + "="*80)
        print("ML-OPTIMIZED V49 BACKTEST")
        print("Strategy: Take ALL signals, size by ML confidence")
        print("="*80)
        
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        print(f"\n📅 Period: {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"💰 Initial: ${self.initial_capital:,.0f}")
        print(f"📊 Position sizing: {self.min_position_size*100:.0f}%-{self.max_position_size*100:.0f}% (ML-based)")
        print(f"🎯 Leverage: {self.leverage}x\n")
        
        signals_taken = 0
        high_conf_trades = 0
        med_conf_trades = 0
        low_conf_trades = 0
        
        for i, date in enumerate(all_dates):
            # Calculate current equity
            total_equity = self.capital
            for symbol, pos in self.positions.items():
                if date in self.data[symbol].index:
                    current_price = float(self.data[symbol].loc[date]['Close'])
                    current_value = pos['shares'] * current_price
                    pnl = current_value - pos['leveraged_value']
                    total_equity += pos['entry_value'] + pnl
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'positions': len(self.positions)
            })
            
            # Check exits
            exits = self.check_exits(date)
            for exit_info in exits:
                self.execute_exit(exit_info)
            
            # Get V49 signals and enhance with ML
            if len(self.positions) < self.max_positions:
                v49_signals = self.get_v49_signals(date)
                
                if v49_signals:
                    # Enhance ALL signals with ML sizing
                    enhanced_signals = self.enhance_signals_with_ml(v49_signals, date)
                    signals_taken += len(enhanced_signals)
                    
                    # Count confidence levels
                    for sig in enhanced_signals:
                        if 'HIGH_CONF' in sig['strategy']:
                            high_conf_trades += 1
                        elif 'LOW_CONF' in sig['strategy']:
                            low_conf_trades += 1
                        else:
                            med_conf_trades += 1
                    
                    # Sort by quality score
                    enhanced_signals = sorted(enhanced_signals, 
                                            key=lambda x: x['quality_score'], 
                                            reverse=True)
                    
                    # Execute trades
                    for signal in enhanced_signals[:self.max_positions - len(self.positions)]:
                        if signal['symbol'] not in self.positions:
                            self.execute_trade(signal)
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Day {i+1}/{len(all_dates)} | "
                      f"Equity: ${total_equity:,.0f} | "
                      f"Signals: {signals_taken} | "
                      f"High: {high_conf_trades} | Med: {med_conf_trades} | Low: {low_conf_trades}")
        
        print(f"\n📊 Signal Summary:")
        print(f"  Total Signals Taken: {signals_taken}")
        print(f"  High Confidence: {high_conf_trades} ({high_conf_trades/max(signals_taken,1)*100:.1f}%)")
        print(f"  Medium Confidence: {med_conf_trades} ({med_conf_trades/max(signals_taken,1)*100:.1f}%)")
        print(f"  Low Confidence: {low_conf_trades} ({low_conf_trades/max(signals_taken,1)*100:.1f}%)")
        
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        if not self.trades:
            print("\n⚠️ No trades executed")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate CAGR
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        cagr = (((final_equity / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        sharpe = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std()
        
        # Max drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_dd = equity_df['drawdown'].min()
        
        # Win/loss stats
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winners) / len(trades_df) * 100
        
        print("\n" + "="*80)
        print("RESULTS: ML-OPTIMIZED V49")
        print("="*80)
        print(f"Initial Capital:    ${self.initial_capital:,.0f}")
        print(f"Final Equity:       ${final_equity:,.0f}")
        print(f"Total Return:       {total_return:.1f}%")
        print(f"🎯 CAGR:            {cagr:.1f}% (Target: 108%+)")
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Max Drawdown:       {max_dd:.1f}%")
        print(f"\nTotal Trades:       {len(trades_df)}")
        print(f"Win Rate:           {win_rate:.1f}%")
        print(f"Avg Win:            ${winners['pnl'].mean():,.0f}" if len(winners) > 0 else "")
        print(f"Avg Loss:           ${losers['pnl'].mean():,.0f}" if len(losers) > 0 else "")
        
        # Analysis by confidence level
        print("\n" + "="*80)
        print("PERFORMANCE BY CONFIDENCE LEVEL")
        print("="*80)
        
        for conf_level in ['HIGH_CONF', 'MED_CONF', 'LOW_CONF']:
            conf_trades = trades_df[trades_df['strategy'].str.contains(conf_level)]
            if len(conf_trades) > 0:
                conf_winners = conf_trades[conf_trades['pnl'] > 0]
                conf_win_rate = len(conf_winners) / len(conf_trades) * 100
                avg_position = conf_trades['position_size'].mean() * 100
                
                print(f"\n{conf_level.replace('_CONF', '')} Confidence Trades ({len(conf_trades)}):")
                print(f"  Win Rate:        {conf_win_rate:.1f}%")
                print(f"  Avg P&L:         ${conf_trades['pnl'].mean():,.0f}")
                print(f"  Avg Position:    {avg_position:.0f}%")
                print(f"  Total P&L:       ${conf_trades['pnl'].sum():,.0f}")
        
        # Save results
        import os
        os.makedirs('ml_optimized_results', exist_ok=True)
        equity_df.to_csv('ml_optimized_results/equity_curve.csv', index=False)
        trades_df.to_csv('ml_optimized_results/trades.csv', index=False)
        
        # Save summary
        with open('ml_optimized_results/summary.txt', 'w') as f:
            f.write(f"ML-OPTIMIZED V49 BACKTEST RESULTS\n")
            f.write(f"="*50 + "\n")
            f.write(f"Period: {equity_df['date'].iloc[0].date()} to {equity_df['date'].iloc[-1].date()}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.0f}\n")
            f.write(f"Final Equity: ${final_equity:,.0f}\n")
            f.write(f"CAGR: {cagr:.1f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write(f"Max Drawdown: {max_dd:.1f}%\n")
            f.write(f"Win Rate: {win_rate:.1f}%\n")
            f.write(f"Total Trades: {len(trades_df)}\n")
        
        print("\n📁 Files saved to ml_optimized_results/")
        print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ML-OPTIMIZED V49 STRATEGY")
    print("Takes ALL V49 signals, sizes positions by ML confidence")
    print("="*80)
    
    # Test with different periods
    import sys
    
    # Default to 2015-2025 if no argument
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2025-01-01'
    
    # Try to load from unified config first
    config = None
    initial_capital = None
    
    if UNIFIED_CONFIG_AVAILABLE and os.path.exists('trading_config.json'):
        try:
            config = TradingConfig()
            print(f"\n✅ Loaded unified trading configuration")
            print(f"   Capital: ${config.total_capital:,.0f}")
            print(f"   Leverage: {config.max_leverage}x")
            if config.positions:
                total_in_positions = sum(p['capital_used'] for p in config.positions.values())
                print(f"   Current Positions: {len(config.positions)} (${total_in_positions:,.0f})")
        except Exception as e:
            print(f"\n⚠️  Could not load unified config: {e}")
            print(f"   Falling back to command line/default")
            config = None
    else:
        if UNIFIED_CONFIG_AVAILABLE:
            print(f"\nℹ️  trading_config.json not found")
        print(f"   Using command line or default capital")
    
    # Fall back to command line or default if no config
    if config is None:
        initial_capital = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
        print(f"\n💰 Capital: ${initial_capital:,.0f} (command line/default)")
    
    # Create backtest with config or capital
    print()
    if config is not None:
        backtest = MLOptimizedV49(config=config)
    else:
        backtest = MLOptimizedV49(initial_capital=initial_capital)
    
    if backtest.load_data(start_date=start_date):
        backtest.run_backtest()
    
    print("\n✅ Complete!")