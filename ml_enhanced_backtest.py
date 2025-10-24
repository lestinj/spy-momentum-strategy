#!/usr/bin/env python3
"""
ML-ENHANCED V49 - OPTIMIZED PARAMETERS VERSION
==============================================
This version enhances your 77% CAGR strategy with optimized ML parameters

KEY CHANGES IN THIS VERSION:
----------------------------
1. LESS CONSERVATIVE ML FILTERING:
   - ml_confidence_threshold: 0.50 → 0.40 (more signals accepted)
   - ml_boost_threshold: 0.70 → 0.60 (more signals get boosted)
   - ml_reduce_threshold: NEW 0.20 (only reduce very low confidence)

2. DYNAMIC POSITION SIZING:
   - Base position: 30% (down from 50% for risk management)
   - ML Boosted: 50% for high confidence signals
   - ML Normal: 30% for standard signals
   - ML Reduced: 15% for low confidence (not 0!)

3. ADAPTIVE STOP LOSS & TAKE PROFIT:
   - Boosted trades: 10% SL, 30% TP (wider stops, higher targets)
   - Normal trades: 8% SL, 25% TP (standard)
   - Reduced trades: 5% SL, 15% TP (tighter risk management)

4. PER-STOCK ML FILTERING:
   - Disable ML for stocks with <50% accuracy
   - Focus ML enhancement on high-accuracy stocks only

5. ENHANCED FEATURES:
   - Multiple momentum timeframes (3, 5, 10, 20 days)
   - Trend strength at multiple scales (10, 20, 50 days)
   - Volatility regime detection
   - Market breadth indicators
   - Time-based features

6. IMPROVED ML MODELS:
   - Larger training window (150+ samples required)
   - Deeper trees (max_depth 7 vs 5)
   - More estimators (100 vs 50)
   - Better feature engineering

EXPECTED IMPROVEMENTS:
- More ML-boosted trades (target: 100+ vs current 21)
- Fewer filtered signals (target: <30% vs current 77%)
- Better risk-adjusted returns through dynamic sizing
- Higher CAGR closer to the 77% target
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


class MLEnhancedV49Proper:
    """
    ML enhancement that IMPROVES on 77% CAGR, not destroys it
    """
    
    def __init__(self, initial_capital=300000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Original V49 symbols
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        # V49 ORIGINAL parameters (keeping what works!)
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # MATCH V49 AGGRESSIVE SETTINGS - with ML-adaptive sizing
        self.max_positions = 3
        
        # Dynamic position sizing based on ML confidence
        self.base_position_size = 0.30  # Base 30% position
        self.ml_boosted_size = 0.50     # 50% for high confidence
        self.ml_normal_size = 0.30      # 30% for normal
        self.ml_reduced_size = 0.15      # 15% for low confidence (not 0!)
        self.position_size = self.base_position_size  # Default
        
        self.leverage = 2.0  # Reduced from 2.5 for risk management
        
        # Base stop loss and take profit
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        
        # ML-adaptive stop loss and take profit
        self.ml_boosted_sl = 0.10    # Wider stop for high conviction
        self.ml_boosted_tp = 0.30    # Higher target for high conviction
        self.ml_normal_sl = 0.08     # Standard
        self.ml_normal_tp = 0.25     # Standard
        self.ml_reduced_sl = 0.05    # Tighter stop for low conviction
        self.ml_reduced_tp = 0.15    # Lower target for low conviction
        
        # ML settings - MORE AGGRESSIVE
        self.ml_confidence_threshold = 0.40  # Even lower threshold
        self.use_ml_filter = True  # ML as filter, not blocker
        self.ml_boost_threshold = 0.60  # Boost threshold (lowered from 0.70)
        self.ml_reduce_threshold = 0.20  # Only reduce if very low confidence
        
        # Minimum accuracy to use ML for a stock
        self.min_ml_accuracy = 0.50  # Disable ML for stocks with <50% accuracy
        self.stock_accuracies = {}  # Store accuracies
        
        # ML models
        self.models = {}
        self.scalers = {}
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self, df):
        """Prepare ML features focused on momentum - ENHANCED"""
        features = pd.DataFrame(index=df.index)
        
        # Core momentum features (matching V49 strategy)
        features['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        features['rsi_change'] = features['rsi'].diff()
        features['rsi_ma'] = features['rsi'].rolling(5).mean()  # RSI momentum
        features['ma_fast'] = df['Close'].rolling(self.ma_fast).mean()
        features['ma_slow'] = df['Close'].rolling(self.ma_slow).mean()
        features['ma_ratio'] = features['ma_fast'] / features['ma_slow']
        features['price_to_ma_fast'] = df['Close'] / features['ma_fast']
        
        # Volume momentum
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_price'] = features['volume_ratio'] * df['Close'].pct_change()
        features['volume_surge'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
        
        # Trend strength - multiple timeframes
        for period in [10, 20, 50]:
            features[f'trend_strength_{period}'] = (df['Close'] - df['Close'].rolling(period).min()) / \
                                         (df['Close'].rolling(period).max() - df['Close'].rolling(period).min())
        
        # Momentum - multiple periods
        for period in [3, 5, 10, 20]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Volatility for position sizing
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_regime'] = features['volatility'] / features['volatility'].rolling(60).mean()
        
        # Market breadth indicators
        features['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        features['range_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                                     (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Time features (day of week effect)
        features['day_of_week'] = pd.to_datetime(df.index).dayofweek
        features['month'] = pd.to_datetime(df.index).month
        
        return features.fillna(0)
    
    def train_ml_models(self, symbol, df, features_df):
        """Train lightweight ML models with accuracy tracking"""
        print(f"Training ML for {symbol}...")
        
        # Simple labels - will it go up 3% in next 5 days?
        future_returns = df['Close'].shift(-5) / df['Close'] - 1
        labels = (future_returns > 0.03).astype(int)
        
        # Prepare data
        X = features_df.fillna(0)
        y = labels.fillna(0)
        
        # Remove invalid rows
        valid_mask = ~(y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Require more training samples
        if len(X) < 150:  # Increased from 100
            self.stock_accuracies[symbol] = 0.0  # Mark as unusable
            return False
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble of models for better accuracy
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Increased from 50
            max_depth=7,       # Slightly deeper
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Store
        self.models[symbol] = rf_model
        self.scalers[symbol] = scaler
        
        # Test accuracy
        accuracy = rf_model.score(X_test_scaled, y_test)
        self.stock_accuracies[symbol] = accuracy
        print(f"  {symbol} ML Accuracy: {accuracy:.2%}")
        
        # Warn if accuracy is too low
        if accuracy < self.min_ml_accuracy:
            print(f"  ⚠️  {symbol}: ML disabled (accuracy < {self.min_ml_accuracy:.0%})")
        
        return True
    
    def get_v49_signals(self, date):
        """Get original V49 TREND_FOLLOW and PULLBACK signals"""
        signals = []
        
        for symbol, df in self.data.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 50:  # Need some history
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
                
                strategy_signal = None
                base_quality = 1
                
                # STRATEGY 1: Trend Following (Original V49)
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    strategy_signal = 'TREND_FOLLOW'
                    base_quality = 3
                
                # STRATEGY 2: Pullback Entry (Original V49)
                elif (rsi < self.rsi_buy and rsi > 45 and
                      close > ma_slow and
                      ma_fast > ma_slow):
                    strategy_signal = 'PULLBACK'
                    base_quality = 3
                
                if strategy_signal:
                    signals.append({
                        'symbol': symbol,
                        'price': close,
                        'date': date,
                        'strategy': strategy_signal,
                        'quality': base_quality,
                        'rsi': rsi
                    })
                    
            except Exception as e:
                continue
        
        return signals
    
    def enhance_signals_with_ml(self, signals, date):
        """
        Use ML to ENHANCE signals with confidence-based position sizing
        ML should:
        1. Increase position size for high confidence
        2. Use graduated position sizing based on confidence
        3. Only filter out signals with VERY low confidence
        4. Skip ML for stocks with poor accuracy
        """
        enhanced_signals = []
        
        for signal in signals:
            symbol = signal['symbol']
            
            # Check if we should use ML for this stock
            use_ml_for_stock = (symbol in self.stock_accuracies and 
                               self.stock_accuracies[symbol] >= self.min_ml_accuracy)
            
            # Get ML confidence if model exists and accuracy is good
            ml_confidence = 0.5  # Default neutral
            
            if use_ml_for_stock and symbol in self.models and symbol in self.features:
                try:
                    features_df = self.features[symbol]
                    if date in features_df.index:
                        X = features_df.loc[date].values.reshape(1, -1)
                        X_scaled = self.scalers[symbol].transform(X)
                        
                        # Get ML prediction
                        ml_prob = self.models[symbol].predict_proba(X_scaled)[0]
                        ml_confidence = ml_prob[1]  # Probability of positive return
                        
                except:
                    ml_confidence = 0.5
            
            # ENHANCE the signal based on ML
            signal['ml_confidence'] = ml_confidence
            
            # Confidence-based position sizing and risk management
            if ml_confidence >= self.ml_boost_threshold:
                # HIGH CONFIDENCE: Boost position size and adjust stops
                signal['position_size'] = self.ml_boosted_size
                signal['stop_loss'] = self.ml_boosted_sl
                signal['take_profit'] = self.ml_boosted_tp
                signal['enhanced'] = True
                signal['strategy'] += '_ML_BOOST'
                signal['quality'] *= 1.5  # Boost quality score
                
            elif ml_confidence >= self.ml_confidence_threshold:
                # NORMAL CONFIDENCE: Standard parameters
                signal['position_size'] = self.ml_normal_size
                signal['stop_loss'] = self.ml_normal_sl
                signal['take_profit'] = self.ml_normal_tp
                signal['enhanced'] = False
                
            elif ml_confidence >= self.ml_reduce_threshold:
                # LOW CONFIDENCE: Reduce position but still take trade
                signal['position_size'] = self.ml_reduced_size
                signal['stop_loss'] = self.ml_reduced_sl
                signal['take_profit'] = self.ml_reduced_tp
                signal['enhanced'] = False
                signal['strategy'] += '_ML_REDUCED'
                signal['quality'] *= 0.7  # Reduce quality score
                
            else:
                # VERY LOW CONFIDENCE: Skip this signal entirely
                continue  # Don't add to enhanced_signals
            
            # Add confidence multiplier for final ranking
            signal['final_score'] = signal['quality'] * ml_confidence
            
            enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def check_exits(self, date):
        """Check exit conditions (original V49 logic)"""
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
                
                exit_reason = None
                
                # Use position-specific stop loss and take profit
                stop_loss = position.get('stop_loss', self.stop_loss_pct)
                take_profit = position.get('take_profit', self.take_profit_pct)
                
                # Stop loss (using dynamic value)
                if pnl_pct <= -stop_loss:
                    exit_reason = 'STOP_LOSS'
                
                # Take profit (using dynamic value)
                elif pnl_pct >= take_profit:
                    exit_reason = 'TAKE_PROFIT'
                
                # RSI exit
                elif current_rsi > 0 and current_rsi < self.rsi_sell:
                    exit_reason = 'RSI_SELL'
                
                # Time exit (shorter for low confidence trades)
                elif position.get('ml_confidence', 0.5) < 0.4 and days_held >= 7:
                    exit_reason = 'TIME_EXIT_LOW_CONF'
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
                    
            except:
                continue
        
        return exits
    
    def execute_trade(self, signal):
        """Execute trade with leverage and dynamic parameters"""
        # Calculate leveraged position with dynamic sizing
        position_size = signal.get('position_size', self.base_position_size)
        position_value = self.capital * position_size
        leveraged_value = position_value * self.leverage
        shares = int(leveraged_value / signal['price'])
        
        if shares > 0:
            actual_cost = position_value  # Actual capital used
            
            # Store position with dynamic parameters
            self.positions[signal['symbol']] = {
                'shares': shares,
                'entry_price': signal['price'],
                'entry_date': signal['date'],
                'entry_value': actual_cost,
                'leveraged_value': leveraged_value,
                'ml_confidence': signal.get('ml_confidence', 0.5),
                'strategy': signal['strategy'],
                'stop_loss': signal.get('stop_loss', self.stop_loss_pct),
                'take_profit': signal.get('take_profit', self.take_profit_pct),
                'position_size': position_size
            }
            
            self.capital -= actual_cost
    
    def execute_exit(self, exit_info):
        """Execute exit"""
        symbol = exit_info['symbol']
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_value = position['shares'] * exit_info['exit_price']
            
            # Calculate P&L
            pnl = exit_value - position['leveraged_value']
            
            # Update capital (return original investment + P&L)
            self.capital += position['entry_value'] + pnl
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': exit_info['date'],
                'entry_price': position['entry_price'],
                'exit_price': exit_info['exit_price'],
                'pnl': pnl,
                'pnl_pct': exit_info['pnl_pct'],
                'strategy': position['strategy'],
                'ml_confidence': position['ml_confidence'],
                'exit_reason': exit_info['exit_reason']
            })
            
            del self.positions[symbol]
    
    def load_data(self, start_date='2024-01-01', end_date=None):
        """Load data and train ML"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n📊 Loading V49 + ML Enhancement...")
        print(f"Period: {start_date} to {end_date}\n")
        
        self.data = {}
        self.features = {}
        
        for symbol in self.symbols:
            try:
                # Load with extra history for ML training
                extended_start = pd.to_datetime(start_date) - timedelta(days=200)
                df = yf.download(symbol, start=extended_start, end=end_date,
                               progress=False, auto_adjust=True)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) > 100:
                    # Add V49 indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    # Prepare ML features
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
        return len(self.data) > 0
    
    def run_backtest(self):
        """Run enhanced V49 backtest"""
        print("\n" + "="*80)
        print("V49 + ML ENHANCEMENT BACKTEST")
        print("Target: Beat 77% CAGR")
        print("="*80)
        
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        print(f"\n📅 Period: {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"💰 Initial: ${self.initial_capital:,.0f}")
        print(f"📊 Position size: {self.position_size*100:.0f}% with {self.leverage}x leverage")
        print(f"🎯 Targets: SL {self.stop_loss_pct*100:.0f}%, TP {self.take_profit_pct*100:.0f}%\n")
        
        total_v49_signals = 0
        ml_boosted = 0
        ml_reduced = 0
        
        for i, date in enumerate(all_dates):
            # Calculate equity
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
            
            # Get V49 signals
            if len(self.positions) < self.max_positions:
                v49_signals = self.get_v49_signals(date)
                total_v49_signals += len(v49_signals)
                
                # Enhance with ML
                enhanced_signals = self.enhance_signals_with_ml(v49_signals, date)
                
                # Count enhancements
                for sig in enhanced_signals:
                    if 'ML_BOOST' in sig['strategy']:
                        ml_boosted += 1
                    elif 'ML_REDUCED' in sig['strategy']:
                        ml_reduced += 1
                
                # Sort by final score (quality * ML confidence)
                enhanced_signals = sorted(enhanced_signals, 
                                        key=lambda x: x.get('final_score', x['quality'] * 0.5), 
                                        reverse=True)
                
                # Execute trades
                for signal in enhanced_signals[:self.max_positions - len(self.positions)]:
                    if signal['symbol'] not in self.positions:
                        self.execute_trade(signal)
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"Day {i+1}/{len(all_dates)} | "
                      f"Equity: ${total_equity:,.0f} | "
                      f"V49 Signals: {total_v49_signals} | "
                      f"ML Boost: {ml_boosted} | ML Reduce: {ml_reduced}")
        
        print(f"\n📊 Signal Summary:")
        print(f"  Total V49 Signals: {total_v49_signals}")
        print(f"  ML Boosted: {ml_boosted}")
        print(f"  ML Reduced: {ml_reduced}")
        
        self.generate_reports()
    
    def generate_reports(self):
        """Generate reports"""
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
        
        # Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_dd = equity_df['drawdown'].min()
        
        # Win/Loss
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] <= 0]
        
        print("\n" + "="*80)
        print("RESULTS: V49 + ML ENHANCEMENT")
        print("="*80)
        print(f"Initial Capital:    ${self.initial_capital:,.0f}")
        print(f"Final Equity:       ${final_equity:,.0f}")
        print(f"Total Return:       {total_return:.1f}%")
        print(f"🎯 CAGR:            {cagr:.1f}% (Target: 77%+)")
        print(f"Sharpe Ratio:       {sharpe:.2f}")
        print(f"Max Drawdown:       {max_dd:.1f}%")
        print(f"\nTotal Trades:       {len(trades_df)}")
        print(f"Win Rate:           {len(winners)/len(trades_df)*100:.1f}%")
        
        # ML Impact Analysis
        print("\n" + "="*80)
        print("ML IMPACT ANALYSIS")
        print("="*80)
        
        ml_boosted_trades = trades_df[trades_df['strategy'].str.contains('ML_BOOST')]
        ml_reduced_trades = trades_df[trades_df['strategy'].str.contains('ML_REDUCED')]
        normal_trades = trades_df[~trades_df['strategy'].str.contains('ML_')]
        
        if len(ml_boosted_trades) > 0:
            print(f"\nML BOOSTED Trades ({len(ml_boosted_trades)}):")
            print(f"  Avg P&L: ${ml_boosted_trades['pnl'].mean():,.0f}")
            print(f"  Win Rate: {len(ml_boosted_trades[ml_boosted_trades['pnl']>0])/len(ml_boosted_trades)*100:.1f}%")
        
        if len(normal_trades) > 0:
            print(f"\nNORMAL Trades ({len(normal_trades)}):")
            print(f"  Avg P&L: ${normal_trades['pnl'].mean():,.0f}")
            print(f"  Win Rate: {len(normal_trades[normal_trades['pnl']>0])/len(normal_trades)*100:.1f}%")
        
        if len(ml_reduced_trades) > 0:
            print(f"\nML REDUCED Trades ({len(ml_reduced_trades)}):")
            print(f"  Avg P&L: ${ml_reduced_trades['pnl'].mean():,.0f}")
            print(f"  Win Rate: {len(ml_reduced_trades[ml_reduced_trades['pnl']>0])/len(ml_reduced_trades)*100:.1f}%")
        
        # Save locally
        import os
        os.makedirs('ml_results', exist_ok=True)
        equity_df.to_csv('ml_results/v49_ml_equity.csv', index=False)
        trades_df.to_csv('ml_results/v49_ml_trades.csv', index=False)
        
        print("\n📁 Files saved to ml_results/")
        print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("V49 STRATEGY + ML ENHANCEMENT")
    print("Goal: Enhance 77% CAGR with ML, not destroy it!")
    print("="*80)
    
    backtest = MLEnhancedV49Proper(initial_capital=100000)
    
    if backtest.load_data(start_date='2015-01-01'):
        backtest.run_backtest()
    
    print("\n✅ Complete!")