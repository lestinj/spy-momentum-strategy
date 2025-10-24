#!/usr/bin/env python3
"""
ML INTEGRATION GUIDE FOR V49 TRADING SYSTEM
===========================================
Practical examples of integrating machine learning
with your existing momentum trading strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import your original system
import sys
sys.path.append('.')  # Add current directory to path

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class MLEnhancedV49:
    """
    Enhanced version of V49 that integrates ML predictions
    with the original TREND_FOLLOW and PULLBACK strategies
    """
    
    def __init__(self, initial_capital=300000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        
        # Original V49 parameters
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # ML components
        self.ml_models = {}
        self.feature_scalers = {}
        
        # Hybrid approach parameters
        self.use_ml_confirmation = True
        self.ml_confidence_threshold = 0.6
        self.ml_weight = 0.5  # Weight given to ML vs traditional signals
        
    def prepare_ml_features(self, df):
        """
        Create a focused set of features that complement your existing strategy
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Core momentum features (aligned with your strategy)
        features['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        features['rsi_7'] = self.calculate_rsi(df['Close'], 7)
        features['rsi_21'] = self.calculate_rsi(df['Close'], 21)
        
        # 2. Moving average features
        features['ma_fast'] = df['Close'].rolling(self.ma_fast).mean()
        features['ma_slow'] = df['Close'].rolling(self.ma_slow).mean()
        features['ma_ratio'] = features['ma_fast'] / features['ma_slow']
        features['price_to_ma_fast'] = df['Close'] / features['ma_fast']
        features['price_to_ma_slow'] = df['Close'] / features['ma_slow']
        
        # 3. Trend strength indicators
        features['ma_slope_fast'] = features['ma_fast'].pct_change(5)
        features['ma_slope_slow'] = features['ma_slow'].pct_change(10)
        features['trend_strength'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # 4. Volume patterns (important for momentum)
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        features['price_volume'] = df['Close'].pct_change() * features['volume_ratio']
        
        # 5. Volatility for risk adjustment
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        features['atr'] = self.calculate_atr(df)
        features['atr_ratio'] = features['atr'] / df['Close']
        
        # 6. Momentum indicators
        features['momentum_5'] = df['Close'].pct_change(5)
        features['momentum_10'] = df['Close'].pct_change(10)
        features['momentum_20'] = df['Close'].pct_change(20)
        
        # 7. Pattern recognition
        features['higher_high'] = (df['High'] > df['High'].rolling(20).max().shift(1)).astype(int)
        features['higher_low'] = (df['Low'] > df['Low'].rolling(20).min().shift(1)).astype(int)
        features['breakout'] = ((df['Close'] > df['High'].rolling(20).max().shift(1)) & 
                                (features['volume_ratio'] > 1.5)).astype(int)
        
        # 8. MACD for trend confirmation
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 9. Bollinger Bands for mean reversion component
        bb_period = 20
        bb_std = df['Close'].rolling(bb_period).std()
        bb_mean = df['Close'].rolling(bb_period).mean()
        features['bb_upper'] = bb_mean + (2 * bb_std)
        features['bb_lower'] = bb_mean - (2 * bb_std)
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def create_training_labels(self, df, lookahead=5, profit_threshold=0.03):
        """
        Create labels for training
        1 = Profitable trade
        0 = Neutral/Loss
        """
        future_returns = (df['Close'].shift(-lookahead) / df['Close']) - 1
        
        # Binary classification: profit vs no profit
        labels = (future_returns > profit_threshold).astype(int)
        
        # Add quality score based on return magnitude
        quality_scores = pd.Series(index=df.index, dtype=float)
        quality_scores[future_returns > 0.10] = 3  # Excellent
        quality_scores[(future_returns > 0.05) & (future_returns <= 0.10)] = 2  # Good
        quality_scores[(future_returns > profit_threshold) & (future_returns <= 0.05)] = 1  # Okay
        quality_scores[future_returns <= profit_threshold] = 0  # Poor
        
        return labels, quality_scores
    
    def train_ml_model(self, symbol, df, features_df):
        """
        Train a lightweight ML model for signal confirmation
        """
        print(f"Training ML model for {symbol}...")
        
        # Create labels
        labels, quality = self.create_training_labels(df)
        
        # Prepare training data
        feature_cols = features_df.columns.tolist()
        X = features_df[feature_cols].fillna(0)
        y = labels.fillna(0)
        
        # Remove invalid samples
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            print(f"  Insufficient data for {symbol}")
            return None
        
        # Split data (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (fast and robust)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Train XGBoost for comparison
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_score = rf_model.score(X_test_scaled, y_test)
        xgb_score = xgb_model.score(X_test_scaled, y_test)
        
        print(f"  RF Accuracy: {rf_score:.3f}, XGB Accuracy: {xgb_score:.3f}")
        
        # Store the better model
        if xgb_score > rf_score:
            best_model = xgb_model
            model_type = 'XGBoost'
        else:
            best_model = rf_model
            model_type = 'RandomForest'
        
        # Store model and scaler
        self.ml_models[symbol] = {
            'model': best_model,
            'scaler': scaler,
            'features': feature_cols,
            'accuracy': max(rf_score, xgb_score),
            'type': model_type
        }
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"  Top 5 features:")
            for _, row in importance.head(5).iterrows():
                print(f"    - {row['feature']}: {row['importance']:.3f}")
        
        return best_model
    
    def get_ml_signal(self, symbol, date, df, features_df):
        """
        Get ML prediction for current date
        """
        if symbol not in self.ml_models or date not in df.index:
            return 0, 0.5  # No signal, neutral confidence
        
        try:
            # Get current features
            feature_cols = self.ml_models[symbol]['features']
            current_features = features_df.loc[date][feature_cols].fillna(0).values.reshape(1, -1)
            
            # Scale features
            scaler = self.ml_models[symbol]['scaler']
            current_scaled = scaler.transform(current_features)
            
            # Get prediction and probability
            model = self.ml_models[symbol]['model']
            prediction = model.predict(current_scaled)[0]
            probability = model.predict_proba(current_scaled)[0]
            
            # Return prediction and confidence
            confidence = max(probability)
            
            return prediction, confidence
            
        except Exception as e:
            return 0, 0.5
    
    def generate_hybrid_signals(self, date):
        """
        Combine traditional V49 signals with ML predictions
        """
        signals = []
        
        for symbol in self.symbols:
            if symbol not in self.data or date not in self.data[symbol].index:
                continue
            
            df = self.data[symbol]
            features_df = self.features[symbol]
            
            idx = df.index.get_loc(date)
            if idx < 50:  # Need history
                continue
            
            try:
                current = df.loc[date]
                
                # Get traditional indicators
                rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                close = float(current['Close'])
                ma_fast = float(current['MA_Fast']) if not pd.isna(current['MA_Fast']) else 0
                ma_slow = float(current['MA_Slow']) if not pd.isna(current['MA_Slow']) else 0
                
                if rsi == 0 or ma_fast == 0 or ma_slow == 0:
                    continue
                
                # Check traditional signals
                traditional_signal = 0
                traditional_strategy = None
                
                # TREND_FOLLOW signal
                if (rsi > self.rsi_buy and close > ma_fast and 
                    close > ma_slow and ma_fast > ma_slow):
                    traditional_signal = 1
                    traditional_strategy = 'TREND_FOLLOW'
                
                # PULLBACK signal
                elif (rsi < self.rsi_buy and rsi > 45 and 
                      close > ma_slow and ma_fast > ma_slow):
                    traditional_signal = 1
                    traditional_strategy = 'PULLBACK'
                
                # Get ML prediction
                ml_signal, ml_confidence = self.get_ml_signal(symbol, date, df, features_df)
                
                # Combine signals
                if self.use_ml_confirmation:
                    # Option 1: ML confirmation required
                    if traditional_signal == 1 and ml_signal == 1 and ml_confidence >= self.ml_confidence_threshold:
                        combined_confidence = (0.5 + ml_confidence) / 1.5  # Average confidence
                        
                        signals.append({
                            'symbol': symbol,
                            'price': close,
                            'date': date,
                            'strategy': f'{traditional_strategy}+ML',
                            'confidence': combined_confidence,
                            'ml_confidence': ml_confidence,
                            'traditional_signal': traditional_signal,
                            'ml_signal': ml_signal,
                            'rsi': rsi
                        })
                    
                    # Option 2: Strong ML signal alone
                    elif ml_signal == 1 and ml_confidence >= 0.75:
                        signals.append({
                            'symbol': symbol,
                            'price': close,
                            'date': date,
                            'strategy': 'ML_ONLY',
                            'confidence': ml_confidence,
                            'ml_confidence': ml_confidence,
                            'traditional_signal': 0,
                            'ml_signal': ml_signal,
                            'rsi': rsi
                        })
                else:
                    # Use weighted combination
                    combined_signal = (self.ml_weight * ml_signal + 
                                     (1 - self.ml_weight) * traditional_signal)
                    
                    if combined_signal >= 0.5:
                        signals.append({
                            'symbol': symbol,
                            'price': close,
                            'date': date,
                            'strategy': traditional_strategy or 'ML',
                            'confidence': ml_confidence if ml_signal else 0.6,
                            'ml_confidence': ml_confidence,
                            'traditional_signal': traditional_signal,
                            'ml_signal': ml_signal,
                            'rsi': rsi
                        })
                        
            except Exception as e:
                continue
        
        # Sort by confidence
        return sorted(signals, key=lambda x: x['confidence'], reverse=True)
    
    def calculate_ml_position_size(self, signal, available_capital):
        """
        Dynamic position sizing based on ML confidence
        """
        base_size = 0.35  # Base position size (35%)
        
        # Adjust based on confidence
        confidence_multiplier = signal['confidence'] ** 2  # Square for more conservative
        
        # Strategy-specific adjustments
        if 'ML' in signal['strategy']:
            if signal['ml_confidence'] >= 0.8:
                strategy_multiplier = 1.2
            elif signal['ml_confidence'] >= 0.7:
                strategy_multiplier = 1.0
            else:
                strategy_multiplier = 0.8
        else:
            strategy_multiplier = 1.0
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * strategy_multiplier
        
        # Cap at 50% of available capital
        max_size = min(0.5, available_capital / self.capital)
        position_size = min(position_size, max_size)
        
        # Minimum size 10%
        return max(0.1, position_size)
    
    def run_enhanced_backtest(self, start_date='2024-01-01', end_date=None):
        """
        Run backtest with ML enhancement
        """
        print("\n" + "="*80)
        print("ML-ENHANCED V49 BACKTEST")
        print("Combining TREND_FOLLOW/PULLBACK with Machine Learning")
        print("="*80)
        
        # Load data
        print("\n1. Loading data...")
        self.data = {}
        self.features = {}
        
        for symbol in self.symbols:
            try:
                # Load extra history for ML training
                extended_start = pd.to_datetime(start_date) - timedelta(days=365)
                df = yf.download(symbol, start=extended_start, end=end_date,
                               progress=False, auto_adjust=True)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) > 200:
                    # Add traditional indicators
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    # Create ML features
                    features_df = self.prepare_ml_features(df)
                    
                    # Train ML model on historical data
                    train_end_idx = df.index.get_loc(pd.to_datetime(start_date))
                    train_df = df.iloc[:train_end_idx]
                    train_features = features_df.iloc[:train_end_idx]
                    
                    if len(train_df) > 100:
                        self.train_ml_model(symbol, train_df, train_features)
                    
                    # Store data for backtest period
                    backtest_start = pd.to_datetime(start_date)
                    self.data[symbol] = df[df.index >= backtest_start]
                    self.features[symbol] = features_df[features_df.index >= backtest_start]
                    
                    print(f"  ✓ {symbol}: Loaded and ML trained")
                    
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        if len(self.data) == 0:
            print("ERROR: No data loaded")
            return
        
        # Run backtest
        print("\n2. Running enhanced backtest...")
        self.equity_curve = []
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        for date in all_dates:
            # Update equity
            total_equity = self.capital
            for symbol, pos in self.positions.items():
                if date in self.data[symbol].index:
                    current_price = float(self.data[symbol].loc[date]['Close'])
                    position_value = pos['shares'] * current_price
                    total_equity += position_value - pos['cost']
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'positions': len(self.positions)
            })
            
            # Check exits (simplified)
            for symbol in list(self.positions.keys()):
                if symbol in self.data and date in self.data[symbol].index:
                    pos = self.positions[symbol]
                    current_price = float(self.data[symbol].loc[date]['Close'])
                    entry_price = pos['entry_price']
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Exit conditions
                    if pnl_pct >= 0.25 or pnl_pct <= -0.08:  # Take profit or stop loss
                        pnl = pos['shares'] * (current_price - entry_price)
                        self.capital += pos['shares'] * current_price
                        
                        self.trades.append({
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'strategy': pos.get('strategy', 'Unknown')
                        })
                        
                        del self.positions[symbol]
            
            # Generate new signals
            if len(self.positions) < 3:  # Max 3 positions
                signals = self.generate_hybrid_signals(date)
                
                for signal in signals[:3 - len(self.positions)]:
                    if signal['symbol'] not in self.positions:
                        # Calculate position size
                        position_size = self.calculate_ml_position_size(signal, self.capital)
                        position_value = self.capital * position_size
                        shares = int(position_value / signal['price'])
                        
                        if shares > 0 and self.capital >= shares * signal['price']:
                            cost = shares * signal['price']
                            self.positions[signal['symbol']] = {
                                'shares': shares,
                                'entry_price': signal['price'],
                                'cost': cost,
                                'strategy': signal['strategy'],
                                'ml_confidence': signal.get('ml_confidence', 0)
                            }
                            self.capital -= cost
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """
        Generate performance report
        """
        if not self.equity_curve:
            print("No data to report")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Initial Capital:  ${self.initial_capital:,.0f}")
        print(f"Final Equity:     ${final_equity:,.0f}")
        print(f"Total Return:     {total_return:.1f}%")
        print(f"Total Trades:     {len(self.trades)}")
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            winners = trades_df[trades_df['pnl'] > 0]
            
            print(f"Win Rate:         {len(winners)/len(trades_df)*100:.1f}%")
            print(f"Avg Win:          ${winners['pnl'].mean():,.0f}" if len(winners) > 0 else "Avg Win: N/A")
            
            # Strategy breakdown
            print("\nStrategy Performance:")
            strategy_stats = trades_df.groupby('strategy').agg({
                'pnl': ['count', 'sum', 'mean']
            })
            print(strategy_stats)
        
        print("="*80)


# Example usage
if __name__ == "__main__":
    print("\nML Enhancement Implementation Guide")
    print("="*50)
    print("\nThis demonstrates how to integrate ML with your V49 system:")
    print("1. Keeps your winning TREND_FOLLOW and PULLBACK strategies")
    print("2. Adds ML confirmation for higher confidence trades")
    print("3. Uses dynamic position sizing based on ML confidence")
    print("4. Provides hybrid signals combining both approaches")
    
    # Run enhanced backtest
    enhanced = MLEnhancedV49(initial_capital=300000)
    enhanced.run_enhanced_backtest(start_date='2024-04-01')
    
    print("\n📌 Key Improvements with ML:")
    print("• Better entry timing with ML confirmation")
    print("• Reduced false signals")
    print("• Dynamic position sizing based on confidence")
    print("• Adaptability to changing market conditions")
    print("• Feature importance insights for strategy refinement")
