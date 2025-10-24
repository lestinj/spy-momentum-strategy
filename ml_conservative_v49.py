#!/usr/bin/env python3
"""
ML-OPTIMIZED V49 - CONSERVATIVE VERSION
========================================
Reduced leverage and position sizes to control drawdowns
Target: 80-100% CAGR with <35% drawdowns
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLConservativeV49:
    """
    Conservative ML enhancement with better risk management
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # V49 symbols
        self.symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET']
        
        # V49 signal parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        
        # CONSERVATIVE POSITION MANAGEMENT
        self.max_positions = 3
        self.leverage = 1.5  # REDUCED from 2.0
        
        # CONSERVATIVE position sizing
        self.base_position_size = 0.25  # REDUCED from 0.30
        
        # TIGHTER SIZING RANGE
        self.min_position_size = 0.10   # Min 10%
        self.max_position_size = 0.35   # REDUCED from 0.50
        
        # Risk parameters
        self.base_stop_loss = 0.08
        self.base_take_profit = 0.20  # REDUCED from 0.25
        
        # ML thresholds
        self.high_confidence = 0.65    # RAISED from 0.60
        self.low_confidence = 0.35     # RAISED from 0.30
        
        # ML settings
        self.min_ml_accuracy = 0.45
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
        """Feature engineering for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Core features
        features['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        features['rsi_change'] = features['rsi'].diff()
        features['ma_fast'] = df['Close'].rolling(self.ma_fast).mean()
        features['ma_slow'] = df['Close'].rolling(self.ma_slow).mean()
        features['ma_ratio'] = features['ma_fast'] / features['ma_slow']
        features['price_to_ma_fast'] = df['Close'] / features['ma_fast']
        
        # Volume
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Trend strength
        for period in [10, 20]:
            features[f'trend_{period}'] = (df['Close'] - df['Close'].rolling(period).min()) / \
                                          (df['Close'].rolling(period).max() - df['Close'].rolling(period).min() + 0.0001)
        
        # Momentum
        for period in [5, 10]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Volatility
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        
        return features.fillna(0)
    
    def train_ml_models(self, symbol, df, features_df):
        """Train ML model"""
        # Simple 3% target in 5 days
        future_returns = df['Close'].shift(-5) / df['Close'] - 1
        labels = (future_returns > 0.03).astype(int)
        
        X = features_df.fillna(0)
        y = labels.fillna(0)
        
        valid_mask = ~(y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 150:
            self.stock_accuracies[symbol] = 0.0
            return False
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=5,       # Shallower
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        accuracy = model.score(X_test_scaled, y_test)
        self.stock_accuracies[symbol] = accuracy
        print(f"  {symbol}: {accuracy:.1%}")
        
        return True
    
    def get_v49_signals(self, date):
        """Get V49 signals"""
        signals = []
        
        for symbol, df in self.data.items():
            if date not in df.index:
                continue
            
            idx = df.index.get_loc(date)
            if idx < 50:
                continue
            
            try:
                current = df.loc[date]
                rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                close = float(current['Close'])
                ma_fast = float(current['MA_Fast']) if not pd.isna(current['MA_Fast']) else 0
                ma_slow = float(current['MA_Slow']) if not pd.isna(current['MA_Slow']) else 0
                
                if rsi == 0 or ma_fast == 0 or ma_slow == 0:
                    continue
                
                # TREND_FOLLOW
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    signals.append({
                        'symbol': symbol,
                        'price': close,
                        'date': date,
                        'strategy': 'TREND',
                        'rsi': rsi
                    })
                
                # PULLBACK
                elif (rsi < self.rsi_buy and rsi > 45 and
                      close > ma_slow and
                      ma_fast > ma_slow):
                    signals.append({
                        'symbol': symbol,
                        'price': close,
                        'date': date,
                        'strategy': 'PULLBACK',
                        'rsi': rsi
                    })
                    
            except:
                continue
        
        return signals
    
    def get_ml_confidence(self, symbol, date):
        """Get ML confidence"""
        if (symbol not in self.stock_accuracies or
            self.stock_accuracies[symbol] < self.min_ml_accuracy or
            symbol not in self.models or
            symbol not in self.features):
            return 0.5
        
        try:
            features_df = self.features[symbol]
            if date not in features_df.index:
                return 0.5
            
            X = features_df.loc[date].values.reshape(1, -1)
            X_scaled = self.scalers[symbol].transform(X)
            ml_prob = self.models[symbol].predict_proba(X_scaled)[0]
            return ml_prob[1]
        except:
            return 0.5
    
    def calculate_position_size(self, ml_confidence):
        """Conservative position sizing"""
        if ml_confidence >= self.high_confidence:
            return self.max_position_size
        elif ml_confidence <= self.low_confidence:
            return self.min_position_size
        else:
            # Linear interpolation
            range_conf = self.high_confidence - self.low_confidence
            range_size = self.max_position_size - self.min_position_size
            normalized = (ml_confidence - self.low_confidence) / range_conf
            return self.min_position_size + (range_size * normalized)
    
    def enhance_signals(self, signals, date):
        """Enhance ALL signals with sizing"""
        enhanced = []
        
        for signal in signals:
            ml_confidence = self.get_ml_confidence(signal['symbol'], date)
            signal['ml_confidence'] = ml_confidence
            signal['position_size'] = self.calculate_position_size(ml_confidence)
            
            # Conservative stops based on position size
            if signal['position_size'] >= 0.30:
                signal['stop_loss'] = 0.10
                signal['take_profit'] = 0.25
            elif signal['position_size'] >= 0.20:
                signal['stop_loss'] = 0.08
                signal['take_profit'] = 0.20
            else:
                signal['stop_loss'] = 0.06
                signal['take_profit'] = 0.15
            
            signal['quality'] = 2 + ml_confidence
            enhanced.append(signal)
        
        return enhanced
    
    def check_exits(self, date):
        """Check exits"""
        exits = []
        
        for symbol, pos in list(self.positions.items()):
            if date not in self.data[symbol].index:
                continue
            
            try:
                current = self.data[symbol].loc[date]
                current_price = float(current['Close'])
                current_rsi = float(current['RSI']) if not pd.isna(current['RSI']) else 0
                
                entry_price = pos['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price
                days_held = (date - pos['entry_date']).days
                
                stop_loss = pos.get('stop_loss', self.base_stop_loss)
                take_profit = pos.get('take_profit', self.base_take_profit)
                
                exit_reason = None
                
                if pnl_pct <= -stop_loss:
                    exit_reason = 'STOP_LOSS'
                elif pnl_pct >= take_profit:
                    exit_reason = 'TAKE_PROFIT'
                elif current_rsi < self.rsi_sell:
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
            except:
                continue
        
        return exits
    
    def execute_trade(self, signal):
        """Execute trade"""
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
                'take_profit': signal.get('take_profit', self.base_take_profit)
            }
            
            self.capital -= position_value
    
    def execute_exit(self, exit_info):
        """Execute exit"""
        symbol = exit_info['symbol']
        if symbol in self.positions:
            pos = self.positions[symbol]
            exit_value = pos['shares'] * exit_info['exit_price']
            pnl = exit_value - pos['leveraged_value']
            actual_return = pos['entry_value'] + pnl
            
            self.trades.append({
                'symbol': symbol,
                'entry_date': pos['entry_date'],
                'exit_date': exit_info['date'],
                'entry_price': pos['entry_price'],
                'exit_price': exit_info['exit_price'],
                'pnl': pnl,
                'pnl_pct': exit_info['pnl_pct']
            })
            
            self.capital += actual_return
            del self.positions[symbol]
    
    def load_data(self, start_date='2015-01-01', end_date=None):
        """Load data"""
        print(f"Loading data from {start_date}...")
        
        self.data = {}
        self.features = {}
        
        for symbol in self.symbols:
            try:
                extended_start = pd.to_datetime(start_date) - timedelta(days=200)
                df = yf.download(symbol, start=extended_start, end=end_date,
                               progress=False, auto_adjust=True)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if len(df) > 100:
                    df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
                    df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
                    df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
                    
                    features = self.prepare_features(df)
                    
                    backtest_start = pd.to_datetime(start_date)
                    train_df = df[df.index < backtest_start]
                    train_features = features[features.index < backtest_start]
                    
                    if len(train_df) > 100:
                        self.train_ml_models(symbol, train_df, train_features)
                    
                    self.data[symbol] = df[df.index >= backtest_start]
                    self.features[symbol] = features[features.index >= backtest_start]
            except:
                pass
        
        return len(self.data) > 0
    
    def run_backtest(self):
        """Run backtest"""
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        for date in all_dates:
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
                'equity': total_equity
            })
            
            # Check exits
            exits = self.check_exits(date)
            for exit_info in exits:
                self.execute_exit(exit_info)
            
            # Get new signals
            if len(self.positions) < self.max_positions:
                v49_signals = self.get_v49_signals(date)
                if v49_signals:
                    enhanced = self.enhance_signals(v49_signals, date)
                    enhanced = sorted(enhanced, key=lambda x: x['quality'], reverse=True)
                    
                    for signal in enhanced[:self.max_positions - len(self.positions)]:
                        if signal['symbol'] not in self.positions:
                            self.execute_trade(signal)
        
        self.generate_report()
    
    def generate_report(self):
        """Generate report"""
        if not self.trades:
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        cagr = (((final_equity / self.initial_capital) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        equity_df['returns'] = equity_df['equity'].pct_change()
        sharpe = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std()
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_dd = equity_df['drawdown'].min()
        
        winners = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winners) / len(trades_df) * 100
        
        print(f"\nRESULTS:")
        print(f"Final Equity: ${final_equity:,.0f}")
        print(f"CAGR: {cagr:.1f}%")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max DD: {max_dd:.1f}%")
        print(f"Win Rate: {win_rate:.1f}%")


if __name__ == "__main__":
    import sys
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2023-01-01'
    
    backtest = MLConservativeV49(initial_capital=100000)
    if backtest.load_data(start_date=start_date):
        backtest.run_backtest()
