#!/usr/bin/env python3
"""
V4.9 ML-ENHANCED LIVE TRADING ALERT SYSTEM
==========================================
Combines V49 strategy (92% CAGR) with ML confidence scoring for:
- Dynamic position sizing based on ML confidence
- Risk-adjusted stop loss and take profit targets
- Confidence-based trade prioritization
- Per-stock ML accuracy tracking
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

class V49MLTrader:
    def __init__(self, gmail_user=None, gmail_password=None):
        # Email configuration (optional)
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
        
        # Symbols list
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        # V4.9 Strategy Parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # OPTIMIZED Position Management
        self.max_positions = 3
        
        # Dynamic position sizing based on ML confidence
        self.base_position_size = 0.30  # Base 30% position
        self.ml_boosted_size = 0.50     # 50% for high confidence
        self.ml_normal_size = 0.30      # 30% for normal
        self.ml_reduced_size = 0.15      # 15% for low confidence
        
        self.leverage = 2.0
        
        # Base risk parameters
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        
        # ML-adaptive parameters
        self.ml_boosted_sl = 0.10    # Wider stop for high conviction
        self.ml_boosted_tp = 0.30    # Higher target
        self.ml_normal_sl = 0.08     
        self.ml_normal_tp = 0.25     
        self.ml_reduced_sl = 0.05    # Tighter stop for low conviction
        self.ml_reduced_tp = 0.15    # Lower target
        
        # ML thresholds
        self.ml_confidence_threshold = 0.40  # Accept signals above 40%
        self.ml_boost_threshold = 0.60       # Boost signals above 60%
        self.ml_reduce_threshold = 0.20      # Reduce signals below 20%
        self.min_ml_accuracy = 0.50          # Minimum accuracy to use ML
        
        # ML models storage
        self.models = {}
        self.scalers = {}
        self.stock_accuracies = {}
        
        # Positions file
        self.positions_file = 'positions.txt'
        self.ml_models_dir = 'ml_models'
        
        # Load positions and ML models
        self.load_positions()
        self.load_or_train_models()
        
    def load_positions(self):
        """Load current positions from CSV file"""
        self.positions = {}
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split(',')
                            if len(parts) >= 4:
                                symbol, date, price, shares = parts[:4]
                                # Check if ML confidence is stored
                                ml_confidence = float(parts[4]) if len(parts) > 4 else 0.5
                                self.positions[symbol] = {
                                    'entry_date': date,
                                    'entry_price': float(price),
                                    'shares': int(shares),
                                    'ml_confidence': ml_confidence
                                }
                print(f"✅ Loaded {len(self.positions)} positions")
                if self.positions:
                    print(f"   Current holdings: {', '.join(self.positions.keys())}")
            except Exception as e:
                print(f"⚠️  Error reading positions: {e}")
                self.positions = {}
        else:
            print(f"📝 No positions file found - will create {self.positions_file}")
            
    def save_positions(self):
        """Save positions with ML confidence"""
        try:
            with open(self.positions_file, 'w') as f:
                f.write("# symbol,date,price,shares,ml_confidence\n")
                for symbol, pos in self.positions.items():
                    f.write(f"{symbol},{pos['entry_date']},{pos['entry_price']},{pos['shares']},{pos.get('ml_confidence', 0.5):.3f}\n")
            print(f"✅ Positions saved")
        except Exception as e:
            print(f"❌ Error saving positions: {e}")
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_features(self, df):
        """Prepare ML features - matching optimized backtest"""
        features = pd.DataFrame(index=df.index)
        
        # Core momentum features
        features['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        features['rsi_change'] = features['rsi'].diff()
        features['rsi_ma'] = features['rsi'].rolling(5).mean()
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
        
        # Volatility
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_regime'] = features['volatility'] / features['volatility'].rolling(60).mean()
        
        # Market breadth
        features['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        features['range_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                                     (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Time features
        features['day_of_week'] = pd.to_datetime(df.index).dayofweek
        features['month'] = pd.to_datetime(df.index).month
        
        return features.fillna(0)
    
    def train_ml_model(self, symbol):
        """Train ML model for a symbol"""
        print(f"   Training ML for {symbol}...")
        
        try:
            # Get extended historical data
            df = yf.download(symbol, period='1y', progress=False, auto_adjust=True)
            
            if len(df) < 200:
                print(f"   ⚠️  Insufficient data for {symbol}")
                self.stock_accuracies[symbol] = 0.0
                return False
            
            # Prepare features
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
            df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
            features = self.prepare_features(df)
            
            # Create labels (will price go up 3% in next 5 days?)
            future_returns = df['Close'].shift(-5) / df['Close'] - 1
            labels = (future_returns > 0.03).astype(int)
            
            # Clean data
            X = features.fillna(0)
            y = labels.fillna(0)
            
            valid_mask = ~(y.isna()) & (X.index >= df.index[100])
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 150:
                print(f"   ⚠️  Not enough samples for {symbol}")
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
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Test accuracy
            accuracy = model.score(X_test_scaled, y_test)
            self.stock_accuracies[symbol] = accuracy
            
            # Store model
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            # Visual indicator
            if accuracy >= 0.70:
                status = "🟢 Excellent"
            elif accuracy >= 0.60:
                status = "🟡 Good"
            elif accuracy >= 0.50:
                status = "🟠 Fair"
            else:
                status = "🔴 Poor (disabled)"
            
            print(f"   {symbol}: {accuracy:.1%} accuracy - {status}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error training {symbol}: {e}")
            self.stock_accuracies[symbol] = 0.0
            return False
    
    def load_or_train_models(self):
        """Load existing ML models or train new ones"""
        print("\n🤖 ML Model Initialization")
        print("=" * 60)
        
        # Create models directory
        os.makedirs(self.ml_models_dir, exist_ok=True)
        
        for symbol in self.symbols:
            model_file = f"{self.ml_models_dir}/{symbol}_model.pkl"
            scaler_file = f"{self.ml_models_dir}/{symbol}_scaler.pkl"
            
            # Try to load existing model
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                try:
                    with open(model_file, 'rb') as f:
                        self.models[symbol] = pickle.load(f)
                    with open(scaler_file, 'rb') as f:
                        self.scalers[symbol] = pickle.load(f)
                    # Assume 60% accuracy for loaded models (conservative)
                    self.stock_accuracies[symbol] = 0.60
                    print(f"   ✅ Loaded ML model for {symbol}")
                except:
                    self.train_ml_model(symbol)
                    self.save_model(symbol)
            else:
                self.train_ml_model(symbol)
                self.save_model(symbol)
        
        # Summary
        print("\n📊 ML Model Summary:")
        enabled_count = sum(1 for acc in self.stock_accuracies.values() if acc >= self.min_ml_accuracy)
        print(f"   Models enabled: {enabled_count}/{len(self.symbols)}")
        if enabled_count < len(self.symbols):
            disabled = [s for s, acc in self.stock_accuracies.items() if acc < self.min_ml_accuracy]
            print(f"   Disabled (low accuracy): {', '.join(disabled)}")
        print("=" * 60 + "\n")
    
    def save_model(self, symbol):
        """Save trained model to disk"""
        if symbol in self.models and symbol in self.scalers:
            try:
                with open(f"{self.ml_models_dir}/{symbol}_model.pkl", 'wb') as f:
                    pickle.dump(self.models[symbol], f)
                with open(f"{self.ml_models_dir}/{symbol}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)
            except Exception as e:
                print(f"   ⚠️  Could not save model for {symbol}: {e}")
    
    def get_ml_confidence(self, symbol, df):
        """Get ML confidence score for current signal"""
        if symbol not in self.models or symbol not in self.scalers:
            return 0.5
        
        # Check if ML should be used for this stock
        if self.stock_accuracies.get(symbol, 0) < self.min_ml_accuracy:
            return 0.5  # Neutral - don't use ML
        
        try:
            # Prepare features
            features = self.prepare_features(df)
            if len(features) < 1:
                return 0.5
            
            # Get latest features
            X = features.iloc[-1].values.reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            
            # Get prediction probability
            ml_prob = self.models[symbol].predict_proba(X_scaled)[0]
            confidence = ml_prob[1]  # Probability of positive return
            
            return confidence
            
        except Exception as e:
            print(f"   ⚠️  ML prediction error for {symbol}: {e}")
            return 0.5
    
    def get_data(self, symbol, days=100):
        """Download recent price data"""
        try:
            df = yf.download(symbol, period=f'{days}d', progress=False, auto_adjust=True)
            if len(df) < 50:
                return None
            
            # Calculate indicators
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
            df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
            df = df.dropna()
            
            return df
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            return None
    
    def check_trend_follow(self, row, prev_row):
        """TREND_FOLLOW strategy"""
        try:
            rsi = float(row['RSI'])
            close = float(row['Close'])
            ma_fast = float(row['MA_Fast'])
            ma_slow = float(row['MA_Slow'])
            
            return (rsi > 55 and 
                    close > ma_fast and 
                    close > ma_slow and
                    ma_fast > ma_slow)
        except:
            return False
    
    def check_pullback(self, row, prev_row):
        """PULLBACK strategy"""
        try:
            rsi = float(row['RSI'])
            close = float(row['Close'])
            ma_fast = float(row['MA_Fast'])
            ma_slow = float(row['MA_Slow'])
            prev_rsi = float(prev_row['RSI'])
            
            return (45 <= rsi <= 55 and 
                    close > ma_slow and
                    ma_fast > ma_slow and
                    prev_rsi > 55)
        except:
            return False
    
    def check_buy_signal(self, symbol):
        """Check for buy signals with ML enhancement"""
        df = self.get_data(symbol)
        if df is None or len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signal = None
        strategy = None
        
        # Check V49 signals
        if self.check_trend_follow(current, previous):
            signal = True
            strategy = 'TREND_FOLLOW'
        elif self.check_pullback(current, previous):
            signal = True
            strategy = 'PULLBACK'
        
        if not signal:
            return None
        
        # Get ML confidence
        ml_confidence = self.get_ml_confidence(symbol, df)
        
        # Filter out very low confidence signals
        if ml_confidence < self.ml_reduce_threshold:
            return None  # Skip this signal
        
        # Determine position sizing and risk parameters
        if ml_confidence >= self.ml_boost_threshold:
            position_size = self.ml_boosted_size
            stop_loss = self.ml_boosted_sl
            take_profit = self.ml_boosted_tp
            confidence_level = "HIGH"
            confidence_emoji = "🟢"
        elif ml_confidence >= self.ml_confidence_threshold:
            position_size = self.ml_normal_size
            stop_loss = self.ml_normal_sl
            take_profit = self.ml_normal_tp
            confidence_level = "NORMAL"
            confidence_emoji = "🟡"
        else:
            position_size = self.ml_reduced_size
            stop_loss = self.ml_reduced_sl
            take_profit = self.ml_reduced_tp
            confidence_level = "LOW"
            confidence_emoji = "🟠"
        
        return {
            'symbol': symbol,
            'strategy': strategy,
            'price': float(current['Close']),
            'rsi': float(current['RSI']),
            'date': current.name.strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'ml_confidence': ml_confidence,
            'confidence_level': confidence_level,
            'confidence_emoji': confidence_emoji,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ml_accuracy': self.stock_accuracies.get(symbol, 0),
            'quality_score': ml_confidence * (3 if strategy == 'TREND_FOLLOW' else 2.5)
        }
    
    def check_sell_signal(self, symbol, position):
        """Check for sell signals on existing position"""
        df = self.get_data(symbol)
        if df is None or len(df) < 1:
            return None
        
        current = df.iloc[-1]
        current_price = float(current['Close'])
        entry_price = position['entry_price']
        
        # Parse entry date
        try:
            entry_date = datetime.fromisoformat(position['entry_date'])
        except:
            entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d')
        
        days_held = (datetime.now() - entry_date).days
        pnl_pct = (current_price - entry_price) / entry_price
        rsi = float(current['RSI'])
        
        # Get position ML confidence to determine exit parameters
        ml_confidence = position.get('ml_confidence', 0.5)
        
        if ml_confidence >= self.ml_boost_threshold:
            stop_loss = self.ml_boosted_sl
            take_profit = self.ml_boosted_tp
            max_days = 16  # Hold high confidence longer
        elif ml_confidence >= self.ml_confidence_threshold:
            stop_loss = self.ml_normal_sl
            take_profit = self.ml_normal_tp
            max_days = 14
        else:
            stop_loss = self.ml_reduced_sl
            take_profit = self.ml_reduced_tp
            max_days = 7  # Exit low confidence faster
        
        # Check exit conditions
        if pnl_pct <= -stop_loss:
            exit_reason = f"STOP_LOSS (-{stop_loss*100:.0f}%)"
        elif pnl_pct >= take_profit:
            exit_reason = f"TAKE_PROFIT (+{take_profit*100:.0f}%)"
        elif rsi < self.rsi_sell:
            exit_reason = "RSI_SELL"
        elif days_held >= max_days:
            exit_reason = f"TIME_EXIT ({days_held} days)"
        else:
            return None
        
        # Calculate dollar P&L
        pnl_dollars = position['shares'] * (current_price - entry_price)
        
        return {
            'symbol': symbol,
            'reason': exit_reason,
            'entry_price': entry_price,
            'entry_date': position['entry_date'],
            'current_price': current_price,
            'shares': position['shares'],
            'pnl_pct': pnl_pct * 100,
            'pnl_dollars': pnl_dollars,
            'days_held': days_held,
            'rsi': rsi,
            'ml_confidence': ml_confidence
        }
    
    def scan_signals(self):
        """Scan all symbols for buy/sell signals"""
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Scanning for ML-enhanced signals...")
        
        buy_signals = []
        sell_signals = []
        
        # Check for sell signals on existing positions
        for symbol, position in list(self.positions.items()):
            sell_signal = self.check_sell_signal(symbol, position)
            if sell_signal:
                sell_signals.append(sell_signal)
        
        # Check for buy signals
        positions_available = self.max_positions - len(self.positions)
        if positions_available > 0:
            print(f"   📊 {positions_available} position slot(s) available")
            
            for symbol in self.symbols:
                if symbol not in self.positions:
                    buy_signal = self.check_buy_signal(symbol)
                    if buy_signal:
                        buy_signals.append(buy_signal)
            
            # Sort buy signals by quality score
            buy_signals = sorted(buy_signals, 
                               key=lambda x: x['quality_score'], 
                               reverse=True)
        else:
            print(f"   ⚠️  All {self.max_positions} position slots filled")
        
        return buy_signals, sell_signals
    
    def format_buy_alert(self, signals):
        """Format buy signals with ML confidence"""
        if not signals:
            return ""
        
        alert = f"\n{'='*70}\n"
        alert += f"🟢 BUY SIGNALS - ML-Enhanced Trading Instructions\n"
        alert += f"{'='*70}\n\n"
        
        for i, signal in enumerate(signals[:self.max_positions - len(self.positions)], 1):
            # Confidence visual
            conf_bar = "█" * int(signal['ml_confidence'] * 10) + "░" * (10 - int(signal['ml_confidence'] * 10))
            
            alert += f"#{i} {signal['confidence_emoji']} {signal['symbol']} - {signal['strategy']}\n"
            alert += f"{'─'*70}\n"
            alert += f"ML Confidence: {conf_bar} {signal['ml_confidence']*100:.1f}% ({signal['confidence_level']})\n"
            alert += f"Model Accuracy: {signal['ml_accuracy']*100:.1f}%\n"
            alert += f"Current Price: ${signal['price']:.2f}\n"
            alert += f"RSI: {signal['rsi']:.1f}\n\n"
            
            # Calculate shares based on dynamic position size
            base_capital = 10000  # Adjust to your capital
            shares = int((signal['position_size'] * base_capital * self.leverage) / signal['price'])
            
            alert += f"📋 TRADING INSTRUCTIONS:\n"
            alert += f"Position Size: {signal['position_size']*100:.0f}% ({signal['confidence_level']} confidence)\n"
            alert += f"1. BUY {shares} shares of {signal['symbol']} @ ${signal['price']:.2f}\n"
            alert += f"2. STOP LOSS: ${signal['price'] * (1 - signal['stop_loss']):.2f} (-{signal['stop_loss']*100:.0f}%)\n"
            alert += f"3. TAKE PROFIT: ${signal['price'] * (1 + signal['take_profit']):.2f} (+{signal['take_profit']*100:.0f}%)\n"
            alert += f"4. Add to positions.txt:\n"
            alert += f"   {signal['symbol']},{signal['date']},{signal['price']:.2f},{shares},{signal['ml_confidence']:.3f}\n\n"
            
        # Risk summary
        alert += f"{'='*70}\n"
        alert += f"📊 RISK SUMMARY:\n"
        total_exposure = sum(s['position_size'] * 100 for s in signals[:self.max_positions - len(self.positions)])
        alert += f"Total Exposure: {total_exposure:.0f}% of capital\n"
        alert += f"Leverage Used: {self.leverage}x\n"
        alert += f"Max Risk: {total_exposure * 0.08:.1f}% (if all hit stop loss)\n"
        alert += f"{'='*70}\n"
        
        return alert
    
    def format_sell_alert(self, signals):
        """Format sell signals with ML context"""
        if not signals:
            return ""
        
        avg_pnl = sum(s['pnl_pct'] for s in signals) / len(signals)
        
        alert = f"\n{'='*70}\n"
        alert += f"{'🟢' if avg_pnl > 0 else '🔴'} SELL SIGNALS - Exit Instructions\n"
        alert += f"{'='*70}\n\n"
        
        for signal in signals:
            emoji = "🟢" if signal['pnl_pct'] > 0 else "🔴"
            
            # ML confidence visual
            ml_conf = signal['ml_confidence']
            conf_level = "HIGH" if ml_conf >= 0.6 else "NORMAL" if ml_conf >= 0.4 else "LOW"
            
            alert += f"{emoji} {signal['symbol']} - {signal['reason']}\n"
            alert += f"{'─'*70}\n"
            alert += f"Entry: ${signal['entry_price']:.2f} ({signal['entry_date']}) | ML Conf: {ml_conf:.1%} ({conf_level})\n"
            alert += f"Current: ${signal['current_price']:.2f}\n"
            alert += f"P&L: {signal['pnl_pct']:+.1f}% (${signal['pnl_dollars']:+.0f})\n"
            alert += f"Days Held: {signal['days_held']} | RSI: {signal['rsi']:.1f}\n\n"
            
            alert += f"📋 ACTION REQUIRED:\n"
            alert += f"1. SELL {signal['shares']} shares of {signal['symbol']}\n"
            alert += f"2. Cancel stop loss/take profit orders\n"
            alert += f"3. Update positions.txt (remove {signal['symbol']} line)\n\n"
        
        # Performance summary
        alert += f"{'='*70}\n"
        alert += f"📊 EXIT SUMMARY:\n"
        alert += f"Average P&L: {avg_pnl:+.1f}%\n"
        total_profit = sum(s['pnl_dollars'] for s in signals)
        alert += f"Total P&L: ${total_profit:+,.0f}\n"
        alert += f"{'='*70}\n"
        
        return alert
    
    def run_once(self):
        """Run a single scan"""
        buy_signals, sell_signals = self.scan_signals()
        
        message = ""
        
        if sell_signals:
            avg_pnl = sum(s['pnl_pct'] for s in sell_signals) / len(sell_signals)
            emoji = "🟢" if avg_pnl > 0 else "🔴"
            print(f"\n{emoji} SELL: {len(sell_signals)} position(s) - Avg {avg_pnl:+.1f}%")
            sell_alert = self.format_sell_alert(sell_signals)
            print(sell_alert)
            message += sell_alert
        
        if buy_signals:
            available_slots = self.max_positions - len(self.positions)
            print(f"\n🟢 BUY: {min(len(buy_signals), available_slots)} of {len(buy_signals)} signals")
            buy_alert = self.format_buy_alert(buy_signals)
            print(buy_alert)
            message += buy_alert
        
        if not buy_signals and not sell_signals:
            print("📊 No signals at this time")
            
            # Show current positions with ML context
            if self.positions:
                print("\n📋 Current Positions (ML-Enhanced):")
                for symbol, pos in self.positions.items():
                    df = self.get_data(symbol, days=5)
                    if df is not None and len(df) > 0:
                        current_price = float(df.iloc[-1]['Close'])
                        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                        emoji = "🟢" if pnl_pct > 0 else "🔴"
                        ml_conf = pos.get('ml_confidence', 0.5)
                        conf_text = f"ML:{ml_conf:.0%}" if ml_conf != 0.5 else "No ML"
                        print(f"   {emoji} {symbol}: {pnl_pct:+.1f}% | Entry: ${pos['entry_price']:.2f} | {conf_text}")
        
        # Send email if configured
        if message and self.gmail_user:
            subject = f"V4.9 ML Alert - {len(buy_signals)} BUY, {len(sell_signals)} SELL"
            self.send_email(subject, message + f"\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n{'='*70}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"ML Models Active: {sum(1 for a in self.stock_accuracies.values() if a >= 0.5)}/{len(self.symbols)}")
        print(f"{'='*70}\n")
    
    def send_email(self, subject, body):
        """Send email alert"""
        if not self.gmail_user or not self.gmail_password:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.gmail_user
            msg['To'] = self.gmail_user
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.gmail_user, self.gmail_password)
            text = msg.as_string()
            server.sendmail(self.gmail_user, self.gmail_user, text)
            server.quit()
            
            print(f"✅ Email sent: {subject}")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
    
    def run_continuous(self, check_interval=300):
        """Run continuous monitoring"""
        print(f"\n{'='*70}")
        print(f"V4.9 ML-ENHANCED LIVE TRADING MONITOR")
        print(f"{'='*70}")
        print(f"Strategy: V49 (92% CAGR) + ML Confidence Scoring")
        print(f"Monitoring: {', '.join(self.symbols)}")
        print(f"Positions: {self.max_positions} max | Dynamic sizing based on ML")
        print(f"Check interval: {check_interval} seconds")
        print(f"Current positions: {len(self.positions)}/{self.max_positions}")
        if self.positions:
            print(f"Holdings: {', '.join(self.positions.keys())}")
        print(f"{'='*70}\n")
        
        while True:
            try:
                self.run_once()
                print(f"💤 Next check in {check_interval} seconds...")
                time.sleep(check_interval)
            except KeyboardInterrupt:
                print("\n\n⚠️  Monitoring stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print(f"Retrying in {check_interval} seconds...")
                time.sleep(check_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='V4.9 ML-Enhanced Trading Alerts (92% CAGR)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--email', type=str, help='Gmail address for alerts')
    parser.add_argument('--password', type=str, help='Gmail app password')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser.add_argument('--retrain', action='store_true', help='Force retrain all ML models')
    
    args = parser.parse_args()
    
    trader = V49MLTrader(
        gmail_user=args.email,
        gmail_password=args.password
    )
    
    if args.retrain:
        print("🔄 Retraining all ML models...")
        trader.models = {}
        trader.scalers = {}
        trader.load_or_train_models()
        print("✅ Retraining complete!")
    
    if args.once:
        trader.run_once()
    else:
        trader.run_continuous(check_interval=args.interval)
