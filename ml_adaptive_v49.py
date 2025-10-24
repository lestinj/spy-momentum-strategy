#!/usr/bin/env python3
"""
ML-ADAPTIVE V49 STRATEGY
========================
Dynamically adjusts leverage and position sizing based on market conditions
Target: Consistent 80-100% CAGR across all market periods

KEY FEATURES:
- Reduces leverage in high volatility
- Increases position size in trending markets
- Adjusts max positions based on correlation
- Adapts stop losses to market regime
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLAdaptiveV49:
    """
    Adaptive ML strategy that adjusts to market conditions
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
        
        # BASE PARAMETERS (will be adjusted dynamically)
        self.base_max_positions = 3
        self.base_leverage = 2.0
        self.base_position_size = 0.30
        
        # ADAPTIVE RANGES
        self.min_leverage = 1.0
        self.max_leverage = 2.5
        self.min_positions = 1
        self.max_positions_allowed = 4
        self.min_position_size = 0.10
        self.max_position_size = 0.50
        
        # Risk parameters
        self.base_stop_loss = 0.08
        self.base_take_profit = 0.25
        
        # Market regime parameters
        self.market_data = {}
        self.market_volatility = 0
        self.market_trend = 0
        self.market_correlation = 0
        self.current_regime = 'NORMAL'
        
        # ML thresholds
        self.high_confidence = 0.60
        self.low_confidence = 0.30
        self.min_ml_accuracy = 0.45
        
        # ML storage
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
    
    def calculate_market_regime(self, date):
        """
        Calculate current market regime to adjust parameters
        """
        try:
            # Get SPY as market proxy
            if 'SPY' not in self.market_data:
                return
            
            spy_df = self.market_data['SPY']
            if date not in spy_df.index:
                return
            
            idx = spy_df.index.get_loc(date)
            if idx < 60:  # Need history
                return
            
            # Calculate market volatility (VIX proxy)
            recent_returns = spy_df['Close'].pct_change().iloc[idx-20:idx]
            self.market_volatility = recent_returns.std() * np.sqrt(252)
            
            # Calculate market trend
            sma_20 = spy_df['Close'].iloc[idx-20:idx].mean()
            sma_50 = spy_df['Close'].iloc[idx-50:idx].mean()
            current_price = spy_df.loc[date, 'Close']
            
            if current_price > sma_20 > sma_50:
                self.market_trend = 1  # Strong uptrend
            elif current_price < sma_20 < sma_50:
                self.market_trend = -1  # Downtrend
            else:
                self.market_trend = 0  # Neutral
            
            # Calculate correlation among positions
            if len(self.positions) > 1:
                returns_data = []
                for symbol in self.positions.keys():
                    if symbol in self.data and idx >= 20:
                        returns = self.data[symbol]['Close'].pct_change().iloc[idx-20:idx]
                        returns_data.append(returns)
                
                if returns_data:
                    returns_df = pd.concat(returns_data, axis=1)
                    corr_matrix = returns_df.corr()
                    # Average correlation
                    self.market_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # Determine regime
            if self.market_volatility > 0.30:  # High volatility (>30% annualized)
                self.current_regime = 'HIGH_VOL'
            elif self.market_volatility < 0.15:  # Low volatility
                self.current_regime = 'LOW_VOL'
            elif self.market_trend == 1:
                self.current_regime = 'TRENDING_UP'
            elif self.market_trend == -1:
                self.current_regime = 'TRENDING_DOWN'
            else:
                self.current_regime = 'NORMAL'
                
        except Exception as e:
            self.current_regime = 'NORMAL'
    
    def get_adaptive_parameters(self):
        """
        Adjust parameters based on market regime
        """
        # Base parameters
        leverage = self.base_leverage
        max_positions = self.base_max_positions
        position_multiplier = 1.0
        stop_loss = self.base_stop_loss
        take_profit = self.base_take_profit
        
        # Adjust based on regime
        if self.current_regime == 'HIGH_VOL':
            # High volatility: Reduce risk
            leverage = max(self.min_leverage, self.base_leverage * 0.6)  # 60% of base
            max_positions = 2
            position_multiplier = 0.7  # Smaller positions
            stop_loss = 0.10  # Wider stops
            take_profit = 0.20  # Closer targets
            
        elif self.current_regime == 'LOW_VOL':
            # Low volatility: Can increase leverage safely
            leverage = min(self.max_leverage, self.base_leverage * 1.2)  # 120% of base
            max_positions = 3
            position_multiplier = 1.1
            stop_loss = 0.06  # Tighter stops
            take_profit = 0.30  # Further targets
            
        elif self.current_regime == 'TRENDING_UP':
            # Strong trend: Maximize opportunity
            leverage = self.base_leverage
            max_positions = 4  # More positions in trend
            position_multiplier = 1.2
            stop_loss = 0.10  # Give room in trend
            take_profit = 0.35  # Let winners run
            
        elif self.current_regime == 'TRENDING_DOWN':
            # Downtrend: Be cautious
            leverage = max(self.min_leverage, self.base_leverage * 0.75)
            max_positions = 2
            position_multiplier = 0.8
            stop_loss = 0.06  # Quick stops
            take_profit = 0.15  # Quick profits
        
        # Adjust for high correlation
        if self.market_correlation > 0.7:
            # High correlation: Reduce positions
            max_positions = max(1, max_positions - 1)
            position_multiplier *= 0.8
        
        return {
            'leverage': leverage,
            'max_positions': max_positions,
            'position_multiplier': position_multiplier,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def prepare_features(self, df):
        """Feature engineering with market regime"""
        features = pd.DataFrame(index=df.index)
        
        # Core features
        features['rsi'] = self.calculate_rsi(df['Close'], self.rsi_period)
        features['rsi_change'] = features['rsi'].diff()
        features['ma_fast'] = df['Close'].rolling(self.ma_fast).mean()
        features['ma_slow'] = df['Close'].rolling(self.ma_slow).mean()
        features['ma_ratio'] = features['ma_fast'] / features['ma_slow']
        features['price_to_ma_fast'] = df['Close'] / features['ma_fast']
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_surge'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5).astype(int)
        
        # Trend strength
        for period in [5, 10, 20]:
            features[f'trend_{period}'] = (df['Close'] - df['Close'].rolling(period).min()) / \
                                          (df['Close'].rolling(period).max() - df['Close'].rolling(period).min() + 0.0001)
        
        # Momentum
        for period in [3, 5, 10]:
            features[f'momentum_{period}'] = df['Close'].pct_change(period)
        
        # Volatility
        features['volatility'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_regime'] = features['volatility'] / features['volatility'].rolling(60).mean()
        
        # Market microstructure
        features['high_low_ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
        features['range_position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                                     (df['High'].rolling(20).max() - df['Low'].rolling(20).min() + 0.0001)
        
        return features.fillna(0)
    
    def train_ml_models(self, symbol, df, features_df):
        """Train ML model"""
        # Target: 3% in 5 days
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
            n_estimators=100,
            max_depth=7,
            min_samples_split=10,
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
                
                signal_type = None
                
                # TREND_FOLLOW
                if (rsi > self.rsi_buy and
                    close > ma_fast and
                    close > ma_slow and
                    ma_fast > ma_slow):
                    signal_type = 'TREND_FOLLOW'
                
                # PULLBACK
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
                    
            except:
                continue
        
        return signals
    
    def get_ml_confidence(self, symbol, date):
        """Get ML confidence"""
        if (symbol not in self.stock_accuracies or
            self.stock_accuracies[symbol] < self.min_ml_accuracy or
            symbol not in self.models):
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
    
    def calculate_dynamic_position_size(self, ml_confidence, position_multiplier):
        """Calculate position size with regime adjustment"""
        # Base calculation
        if ml_confidence >= self.high_confidence:
            base_size = self.max_position_size
        elif ml_confidence <= self.low_confidence:
            base_size = self.min_position_size
        else:
            # Linear interpolation
            range_conf = self.high_confidence - self.low_confidence
            range_size = self.max_position_size - self.min_position_size
            normalized = (ml_confidence - self.low_confidence) / range_conf
            base_size = self.min_position_size + (range_size * normalized)
        
        # Apply regime multiplier
        adjusted_size = base_size * position_multiplier
        
        # Ensure within bounds
        return max(self.min_position_size, min(self.max_position_size, adjusted_size))
    
    def enhance_signals_with_ml(self, signals, date, adaptive_params):
        """Enhance signals with ML and adaptive parameters"""
        enhanced = []
        
        for signal in signals:
            ml_confidence = self.get_ml_confidence(signal['symbol'], date)
            signal['ml_confidence'] = ml_confidence
            
            # Calculate position size with regime adjustment
            position_size = self.calculate_dynamic_position_size(
                ml_confidence, 
                adaptive_params['position_multiplier']
            )
            signal['position_size'] = position_size
            
            # Use adaptive stops
            signal['stop_loss'] = adaptive_params['stop_loss']
            signal['take_profit'] = adaptive_params['take_profit']
            
            # Adjust stops further based on position size
            if position_size >= 0.40:
                signal['stop_loss'] *= 1.2  # Wider stop for large position
            elif position_size <= 0.15:
                signal['stop_loss'] *= 0.8  # Tighter stop for small position
            
            # Quality score
            regime_bonus = 0
            if self.current_regime == 'TRENDING_UP':
                regime_bonus = 0.5
            elif self.current_regime == 'HIGH_VOL':
                regime_bonus = -0.3
            
            signal['quality'] = (2 + ml_confidence + regime_bonus) * \
                               (1.2 if signal['strategy'] == 'TREND_FOLLOW' else 1.0)
            
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
                
                # Dynamic exit based on regime
                if self.current_regime == 'HIGH_VOL' and days_held > 5:
                    # Quick exit in high volatility
                    if pnl_pct > 0.10:  # Take 10% profit quickly
                        exit_reason = 'REGIME_PROFIT'
                
                if not exit_reason:
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
    
    def execute_trade(self, signal, leverage):
        """Execute trade with adaptive leverage"""
        position_size = signal.get('position_size', self.base_position_size)
        position_value = self.capital * position_size
        leveraged_value = position_value * leverage  # Use adaptive leverage
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
                'regime': self.current_regime
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
                'pnl_pct': exit_info['pnl_pct'],
                'regime': pos.get('regime', 'UNKNOWN'),
                'exit_reason': exit_info['exit_reason']
            })
            
            self.capital += actual_return
            del self.positions[symbol]
    
    def load_data(self, start_date='2015-01-01', end_date=None):
        """Load data including market data"""
        print(f"Loading adaptive strategy data from {start_date}...")
        
        # Load SPY for market regime
        try:
            extended_start = pd.to_datetime(start_date) - timedelta(days=200)
            spy = yf.download('SPY', start=extended_start, end=end_date,
                            progress=False, auto_adjust=True)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            self.market_data['SPY'] = spy
            print("  SPY market data loaded")
        except:
            print("  Warning: Could not load SPY market data")
        
        self.data = {}
        self.features = {}
        
        for symbol in self.symbols:
            try:
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
        """Run adaptive backtest"""
        print("\nRunning Adaptive ML Backtest...")
        
        all_dates = sorted(set().union(*[df.index for df in self.data.values()]))
        
        regime_counts = {'HIGH_VOL': 0, 'LOW_VOL': 0, 'TRENDING_UP': 0, 
                        'TRENDING_DOWN': 0, 'NORMAL': 0}
        
        for i, date in enumerate(all_dates):
            # Calculate market regime
            self.calculate_market_regime(date)
            regime_counts[self.current_regime] = regime_counts.get(self.current_regime, 0) + 1
            
            # Get adaptive parameters
            adaptive_params = self.get_adaptive_parameters()
            
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
                'regime': self.current_regime,
                'leverage': adaptive_params['leverage'],
                'positions': len(self.positions)
            })
            
            # Check exits
            exits = self.check_exits(date)
            for exit_info in exits:
                self.execute_exit(exit_info)
            
            # Get new signals
            if len(self.positions) < adaptive_params['max_positions']:
                v49_signals = self.get_v49_signals(date)
                if v49_signals:
                    enhanced = self.enhance_signals_with_ml(v49_signals, date, adaptive_params)
                    enhanced = sorted(enhanced, key=lambda x: x['quality'], reverse=True)
                    
                    for signal in enhanced[:adaptive_params['max_positions'] - len(self.positions)]:
                        if signal['symbol'] not in self.positions:
                            self.execute_trade(signal, adaptive_params['leverage'])
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"Day {i+1}/{len(all_dates)} | "
                      f"Equity: ${total_equity:,.0f} | "
                      f"Regime: {self.current_regime} | "
                      f"Leverage: {adaptive_params['leverage']:.1f}x")
        
        print(f"\nRegime Distribution:")
        for regime, count in regime_counts.items():
            if count > 0:
                print(f"  {regime}: {count/len(all_dates)*100:.1f}%")
        
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive report"""
        if not self.trades:
            print("\nNo trades executed")
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
        
        print("\n" + "="*80)
        print("ADAPTIVE ML RESULTS")
        print("="*80)
        print(f"Final Equity: ${final_equity:,.0f}")
        print(f"CAGR: {cagr:.1f}%")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max DD: {max_dd:.1f}%")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Trades: {len(trades_df)}")
        
        # Performance by regime
        print("\nPerformance by Market Regime:")
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            if len(regime_trades) > 0:
                regime_winners = regime_trades[regime_trades['pnl'] > 0]
                regime_wr = len(regime_winners) / len(regime_trades) * 100
                avg_pnl = regime_trades['pnl'].mean()
                print(f"  {regime}: {len(regime_trades)} trades, "
                      f"{regime_wr:.1f}% win rate, ${avg_pnl:,.0f} avg P&L")


if __name__ == "__main__":
    import sys
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
    
    backtest = MLAdaptiveV49(initial_capital=16000)
    if backtest.load_data(start_date=start_date):
        backtest.run_backtest()
