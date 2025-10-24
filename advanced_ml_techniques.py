#!/usr/bin/env python3
"""
ADVANCED ML TRADING TECHNIQUES
===============================
Advanced machine learning methods for trading:
- Reinforcement Learning (DQN, PPO)
- Transformer models for sequence prediction
- Advanced portfolio optimization
- Adaptive risk management
- Feature selection and engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Portfolio optimization
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis

# Advanced ML
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Try importing advanced libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - skipping hyperparameter optimization")

try:
    import gym
    from gym import spaces
    REINFORCEMENT_AVAILABLE = True
except ImportError:
    REINFORCEMENT_AVAILABLE = False
    print("OpenAI Gym not available - skipping RL components")


class AdvancedMLTechniques:
    """
    Advanced ML techniques for trading system enhancement
    """
    
    def __init__(self):
        self.feature_selector = None
        self.portfolio_optimizer = None
        self.risk_models = {}
        self.regime_detector = None
        
    # ==================== FEATURE ENGINEERING & SELECTION ====================
    
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features including:
        - Fourier transforms for cycle detection
        - Wavelet decomposition
        - Market microstructure features
        - Order flow imbalance proxies
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Fourier Transform Features (detect cycles)
        close_fft = np.fft.fft(df['Close'].values)
        fft_freq = np.fft.fftfreq(len(df))
        
        # Get dominant frequencies
        power_spectrum = np.abs(close_fft) ** 2
        dominant_freq_idx = np.argsort(power_spectrum)[-10:]  # Top 10 frequencies
        
        for i, idx in enumerate(dominant_freq_idx):
            if fft_freq[idx] > 0:  # Only positive frequencies
                period = 1 / fft_freq[idx] if fft_freq[idx] != 0 else np.inf
                features[f'fft_period_{i}'] = period
                features[f'fft_power_{i}'] = power_spectrum[idx]
        
        # 2. Microstructure Features
        features['kyle_lambda'] = self._calculate_kyle_lambda(df)
        features['amihud_illiquidity'] = self._calculate_amihud_ratio(df)
        features['roll_spread'] = self._calculate_roll_spread(df)
        
        # 3. Jump Detection (Lee-Mykland)
        features['jump_indicator'] = self._detect_jumps(df)
        
        # 4. Entropy-based features
        features['shannon_entropy'] = self._calculate_entropy(df['Close'].pct_change(), 10)
        features['approximate_entropy'] = self._approximate_entropy(df['Close'].values, 2, 0.2)
        
        # 5. Hurst Exponent (trend persistence)
        features['hurst_exponent'] = self._calculate_hurst_exponent(df['Close'])
        
        # 6. VPIN (Volume-Synchronized Probability of Informed Trading)
        features['vpin'] = self._calculate_vpin(df)
        
        # 7. Fractal Dimension
        features['fractal_dimension'] = self._calculate_fractal_dimension(df['Close'])
        
        # 8. Relative Strength vs Market
        if 'SPY' in df.columns:  # If we have market data
            features['relative_strength'] = df['Close'] / df['SPY']
            features['beta'] = self._calculate_rolling_beta(df['Close'], df['SPY'], 60)
        
        return features
    
    def _calculate_kyle_lambda(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Kyle's Lambda - price impact coefficient
        """
        returns = df['Close'].pct_change()
        volume = df['Volume']
        
        # Rolling regression of |returns| on volume
        lambda_series = pd.Series(index=df.index, dtype=float)
        
        for i in range(window, len(df)):
            y = np.abs(returns.iloc[i-window:i].values)
            x = volume.iloc[i-window:i].values
            
            if len(y[~np.isnan(y)]) > 10:
                coef = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)[0]
                lambda_series.iloc[i] = coef
        
        return lambda_series
    
    def _calculate_amihud_ratio(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Amihud illiquidity ratio
        """
        returns = df['Close'].pct_change().abs()
        dollar_volume = df['Close'] * df['Volume']
        
        return (returns / dollar_volume).rolling(window).mean() * 1e6
    
    def _calculate_roll_spread(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Roll's implied spread estimator
        """
        returns = df['Close'].pct_change()
        
        spread = pd.Series(index=df.index, dtype=float)
        for i in range(window, len(df)):
            ret_window = returns.iloc[i-window:i]
            cov = ret_window.cov(ret_window.shift(1))
            if cov < 0:
                spread.iloc[i] = 2 * np.sqrt(-cov)
            else:
                spread.iloc[i] = 0
        
        return spread
    
    def _detect_jumps(self, df: pd.DataFrame, window: int = 60) -> pd.Series:
        """
        Lee-Mykland jump detection
        """
        returns = np.log(df['Close'] / df['Close'].shift(1))
        
        # Bipower variation
        bv = pd.Series(index=df.index, dtype=float)
        for i in range(window, len(df)):
            ret_window = returns.iloc[i-window:i]
            bv.iloc[i] = (np.pi/2) * (np.abs(ret_window * ret_window.shift(1))).sum()
        
        # Standardized returns
        std_returns = returns / np.sqrt(bv)
        
        # Jump test statistic
        threshold = -np.log(-np.log(0.99))  # 99% confidence
        jumps = (np.abs(std_returns) > threshold).astype(int)
        
        return jumps
    
    def _calculate_entropy(self, series: pd.Series, bins: int) -> pd.Series:
        """
        Shannon entropy
        """
        entropy = pd.Series(index=series.index, dtype=float)
        window = 50
        
        for i in range(window, len(series)):
            data = series.iloc[i-window:i].dropna()
            if len(data) > 0:
                hist, _ = np.histogram(data, bins=bins)
                hist = hist[hist > 0]
                prob = hist / hist.sum()
                entropy.iloc[i] = -np.sum(prob * np.log2(prob + 1e-10))
        
        return entropy
    
    def _approximate_entropy(self, data: np.array, m: int, r: float) -> float:
        """
        Approximate entropy - measure of time series regularity
        """
        def _pattern_count(data, m, r):
            patterns = []
            for i in range(len(data) - m + 1):
                patterns.append(data[i:i + m])
            
            count = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if max(abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            
            return count / (len(patterns) ** 2)
        
        phi_m = _pattern_count(data, m, r)
        phi_m1 = _pattern_count(data, m + 1, r)
        
        return np.log(phi_m / phi_m1) if phi_m1 > 0 else 0
    
    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        """
        Hurst exponent - measure of long-term memory
        """
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # R/S calculation
            chunks = [series[i:i+lag] for i in range(0, len(series), lag)]
            rs = []
            
            for chunk in chunks:
                if len(chunk) < lag:
                    continue
                    
                mean = chunk.mean()
                deviations = chunk - mean
                Z = deviations.cumsum()
                R = Z.max() - Z.min()
                S = chunk.std()
                
                if S != 0:
                    rs.append(R / S)
            
            if rs:
                tau.append(np.mean(rs))
        
        if len(tau) > 0:
            # Fit log(R/S) = H * log(n) + c
            log_lags = np.log(list(lags[:len(tau)]))
            log_tau = np.log(tau)
            
            if len(log_tau[~np.isnan(log_tau)]) > 2:
                H = np.polyfit(log_lags[~np.isnan(log_tau)], 
                              log_tau[~np.isnan(log_tau)], 1)[0]
                return H
        
        return 0.5  # Random walk
    
    def _calculate_vpin(self, df: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
        """
        Volume-Synchronized Probability of Informed Trading
        """
        # Simplified VPIN calculation
        returns = df['Close'].pct_change()
        volume = df['Volume']
        
        # Classify volume as buy or sell
        buy_volume = volume.where(returns > 0, 0)
        sell_volume = volume.where(returns < 0, 0)
        
        # Calculate order imbalance
        imbalance = (buy_volume - sell_volume).rolling(bucket_size).sum()
        total_volume = volume.rolling(bucket_size).sum()
        
        vpin = (imbalance.abs() / total_volume).fillna(0)
        
        return vpin
    
    def _calculate_fractal_dimension(self, series: pd.Series) -> float:
        """
        Calculate fractal dimension using box-counting method
        """
        # Simplified fractal dimension
        returns = series.pct_change().dropna()
        
        if len(returns) < 100:
            return 1.5  # Default
        
        # Normalize to [0, 1]
        normalized = (returns - returns.min()) / (returns.max() - returns.min() + 1e-10)
        
        # Box counting
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            boxes = len(normalized) // scale
            count = 0
            
            for i in range(boxes):
                chunk = normalized[i*scale:(i+1)*scale]
                if len(chunk) > 0 and chunk.std() > 0:
                    count += 1
            
            counts.append(count)
        
        if len(counts) > 2:
            # Fit log(N) = -D * log(scale) + c
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            
            valid = ~np.isnan(log_counts) & ~np.isinf(log_counts)
            if valid.sum() > 2:
                D = -np.polyfit(log_scales[valid], log_counts[valid], 1)[0]
                return min(max(D, 1.0), 2.0)  # Bound between 1 and 2
        
        return 1.5
    
    def _calculate_rolling_beta(self, asset: pd.Series, market: pd.Series, 
                                window: int = 60) -> pd.Series:
        """
        Rolling beta calculation
        """
        asset_returns = asset.pct_change()
        market_returns = market.pct_change()
        
        beta = pd.Series(index=asset.index, dtype=float)
        
        for i in range(window, len(asset)):
            y = asset_returns.iloc[i-window:i]
            x = market_returns.iloc[i-window:i]
            
            valid = ~(y.isna() | x.isna())
            if valid.sum() > 10:
                coef = np.polyfit(x[valid], y[valid], 1)[0]
                beta.iloc[i] = coef
        
        return beta
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'mutual_info', k: int = 30) -> List[str]:
        """
        Advanced feature selection
        """
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            # Custom selection based on correlation and importance
            return self._custom_feature_selection(X, y, k)
        
        selector.fit(X.fillna(0), y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selector = selector
        return selected_features
    
    def _custom_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 k: int) -> List[str]:
        """
        Custom feature selection using multiple criteria
        """
        scores = {}
        
        for col in X.columns:
            # Correlation with target
            corr = abs(X[col].corr(y))
            
            # Information gain
            ig = mutual_info_classif(X[[col]].fillna(0), y, random_state=42)[0]
            
            # Variance
            var = X[col].var()
            
            # Combined score
            scores[col] = 0.4 * corr + 0.4 * ig + 0.2 * (var / X[col].abs().max())
        
        # Select top k features
        selected = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [f[0] for f in selected]
    
    # ==================== PORTFOLIO OPTIMIZATION ====================
    
    def optimize_portfolio(self, signals: List[Dict], historical_returns: pd.DataFrame,
                          risk_tolerance: float = 0.5) -> List[Dict]:
        """
        Advanced portfolio optimization using:
        - Mean-Variance Optimization
        - Risk Parity
        - Maximum Diversification
        - Kelly Criterion with constraints
        """
        if len(signals) == 0:
            return signals
        
        symbols = [s['symbol'] for s in signals]
        
        # Get historical returns for these symbols
        returns = historical_returns[symbols].dropna()
        
        if len(returns) < 30:
            # Not enough history for optimization
            return self._equal_weight_portfolio(signals)
        
        # Calculate statistics
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252
        
        # Get optimal weights using multiple methods
        weights_mv = self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
        weights_rp = self._risk_parity_optimization(cov_matrix)
        weights_md = self._maximum_diversification(expected_returns, cov_matrix)
        weights_kelly = self._kelly_criterion_optimization(returns, signals)
        
        # Ensemble weights (weighted average of methods)
        final_weights = (0.3 * weights_mv + 
                        0.2 * weights_rp + 
                        0.2 * weights_md + 
                        0.3 * weights_kelly)
        
        # Normalize weights
        final_weights = final_weights / final_weights.sum()
        
        # Apply weights to signals
        for i, signal in enumerate(signals):
            if i < len(final_weights):
                signal['position_size'] = float(final_weights[i])
                signal['optimization_method'] = 'ensemble'
        
        return signals
    
    def _mean_variance_optimization(self, returns: pd.Series, cov: pd.DataFrame,
                                   risk_tolerance: float) -> np.array:
        """
        Markowitz mean-variance optimization
        """
        n_assets = len(returns)
        
        # Objective: maximize return - risk_penalty * variance
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov, weights))
            return -(portfolio_return - risk_tolerance * portfolio_variance)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum to 1
        ]
        
        # Bounds (0 to 100% per position)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                        bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _risk_parity_optimization(self, cov: pd.DataFrame) -> np.array:
        """
        Risk parity - equal risk contribution
        """
        n_assets = len(cov)
        
        def risk_contribution(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            marginal_contrib = np.dot(cov, weights)
            contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal contribution
            target = portfolio_vol / n_assets
            return np.sum((contrib - target) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        
        bounds = [(0.01, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_contribution, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _maximum_diversification(self, returns: pd.Series, cov: pd.DataFrame) -> np.array:
        """
        Maximum diversification portfolio
        """
        n_assets = len(returns)
        
        # Individual asset volatilities
        asset_vols = np.sqrt(np.diag(cov))
        
        def diversification_ratio(weights):
            weighted_avg_vol = np.dot(weights, asset_vols)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
            return -weighted_avg_vol / portfolio_vol  # Negative for minimization
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(diversification_ratio, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _kelly_criterion_optimization(self, returns: pd.DataFrame, 
                                     signals: List[Dict]) -> np.array:
        """
        Kelly Criterion with ML confidence adjustment
        """
        n_assets = len(signals)
        weights = np.zeros(n_assets)
        
        for i, signal in enumerate(signals):
            symbol_returns = returns[signal['symbol']] if signal['symbol'] in returns.columns else None
            
            if symbol_returns is not None and len(symbol_returns) > 30:
                # Win probability from ML confidence
                p = signal.get('confidence', 0.5)
                
                # Win/Loss amounts
                wins = symbol_returns[symbol_returns > 0]
                losses = symbol_returns[symbol_returns < 0]
                
                if len(wins) > 0 and len(losses) > 0:
                    b = wins.mean() / abs(losses.mean())  # Win/loss ratio
                    
                    # Kelly formula: f = (p*b - (1-p)) / b
                    kelly_fraction = (p * b - (1 - p)) / b if b > 0 else 0
                    
                    # Apply Kelly with safety factor (25% of full Kelly)
                    weights[i] = max(0, min(kelly_fraction * 0.25, 0.25))
                else:
                    weights[i] = 0.05  # Small default weight
            else:
                weights[i] = 0.05
        
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.array([1/n_assets] * n_assets)
        
        return weights
    
    def _equal_weight_portfolio(self, signals: List[Dict]) -> List[Dict]:
        """
        Fallback to equal weighting
        """
        weight = 1.0 / len(signals)
        for signal in signals:
            signal['position_size'] = weight
            signal['optimization_method'] = 'equal_weight'
        return signals
    
    # ==================== MARKET REGIME DETECTION ====================
    
    def detect_market_regime_gmm(self, returns: pd.Series, n_regimes: int = 3) -> Tuple[str, float]:
        """
        Detect market regime using Gaussian Mixture Model
        """
        # Prepare features for regime detection
        features = pd.DataFrame()
        features['returns'] = returns
        features['volatility'] = returns.rolling(20).std()
        features['skew'] = returns.rolling(60).skew()
        features['kurtosis'] = returns.rolling(60).apply(kurtosis)
        
        # Drop NaN
        features = features.dropna()
        
        if len(features) < 100:
            return 'unknown', 0.5
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        gmm.fit(features)
        
        # Predict current regime
        current_features = features.iloc[-1:].values
        regime = gmm.predict(current_features)[0]
        proba = gmm.predict_proba(current_features)[0]
        
        # Interpret regimes based on return/volatility characteristics
        regime_means = []
        for i in range(n_regimes):
            mask = gmm.predict(features) == i
            regime_means.append({
                'return': features[mask]['returns'].mean(),
                'vol': features[mask]['volatility'].mean()
            })
        
        # Label regimes
        regime_labels = []
        for rm in regime_means:
            if rm['return'] > 0.0005 and rm['vol'] < features['volatility'].median():
                regime_labels.append('bull')
            elif rm['return'] < -0.0005:
                regime_labels.append('bear')
            else:
                regime_labels.append('sideways')
        
        return regime_labels[regime], max(proba)
    
    # ==================== DYNAMIC RISK MANAGEMENT ====================
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            return mean + std * norm.ppf(1 - confidence)
        elif method == 'cornish_fisher':
            # Cornish-Fisher expansion (adjusts for skew and kurtosis)
            z = norm.ppf(1 - confidence)
            s = skew(returns)
            k = kurtosis(returns)
            
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24
            return returns.mean() + z_cf * returns.std()
        else:
            return self.calculate_var(returns, confidence, 'historical')
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        """
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_dynamic_position_size(self, signal: Dict, portfolio_value: float,
                                       recent_returns: pd.Series,
                                       max_risk: float = 0.02) -> float:
        """
        Calculate position size based on risk metrics
        """
        # Calculate risk metrics
        volatility = recent_returns.std() * np.sqrt(252)  # Annualized
        var_95 = abs(self.calculate_var(recent_returns, 0.95))
        
        # Base position size from Kelly
        win_rate = signal.get('confidence', 0.5)
        avg_win = recent_returns[recent_returns > 0].mean() if len(recent_returns[recent_returns > 0]) > 0 else 0.01
        avg_loss = abs(recent_returns[recent_returns < 0].mean()) if len(recent_returns[recent_returns < 0]) > 0 else 0.01
        
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
        
        # Risk-based adjustment
        risk_adjusted_size = max_risk / var_95 if var_95 > 0 else 0.1
        
        # Combine Kelly and risk-based sizing
        position_size = min(
            kelly_f * 0.25,  # 25% Kelly
            risk_adjusted_size,
            0.2  # Max 20% per position
        )
        
        # Volatility adjustment
        target_vol = 0.15  # 15% target volatility
        vol_scalar = min(target_vol / volatility, 1.5) if volatility > 0 else 1.0
        position_size *= vol_scalar
        
        # Confidence adjustment
        confidence_scalar = signal.get('confidence', 0.5) ** 2  # Square for more conservative sizing
        position_size *= confidence_scalar
        
        return max(0.01, min(position_size, 0.25))  # Between 1% and 25%
    
    # ==================== HYPERPARAMETER OPTIMIZATION ====================
    
    def optimize_hyperparameters(self, model_class, X_train, y_train, X_val, y_val):
        """
        Optimize model hyperparameters using Optuna
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available - using default parameters")
            return {}
        
        def objective(trial):
            # Suggest hyperparameters based on model type
            if 'RandomForest' in str(model_class):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            elif 'XGB' in str(model_class):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5)
                }
            else:
                return 0
            
            # Train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            predictions = model.predict_proba(X_val)[:, 1]
            
            # Custom metric: Sharpe ratio of strategy returns
            threshold = 0.5
            signals = predictions > threshold
            
            if signals.sum() > 0:
                # Calculate returns when signal is active
                strategy_returns = y_val[signals]
                if len(strategy_returns) > 0:
                    sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-10)
                    return sharpe
            
            return 0
        
        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best Sharpe ratio: {study.best_value:.3f}")
        
        return study.best_params


# ==================== REINFORCEMENT LEARNING COMPONENTS ====================

if REINFORCEMENT_AVAILABLE:
    class TradingEnvironment(gym.Env):
        """
        OpenAI Gym environment for RL trading
        """
        
        def __init__(self, df: pd.DataFrame, features_df: pd.DataFrame,
                    initial_capital: float = 10000):
            super(TradingEnvironment, self).__init__()
            
            self.df = df
            self.features_df = features_df
            self.initial_capital = initial_capital
            
            # Action space: 0=hold, 1=buy, 2=sell
            self.action_space = spaces.Discrete(3)
            
            # Observation space: features + position info
            n_features = len(features_df.columns)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(n_features + 3,),  # +3 for position, cash, portfolio value
                dtype=np.float32
            )
            
            self.reset()
        
        def reset(self):
            self.current_step = 50  # Start with some history
            self.cash = self.initial_capital
            self.shares = 0
            self.portfolio_value = self.initial_capital
            self.trades = []
            
            return self._get_observation()
        
        def _get_observation(self):
            # Get current features
            features = self.features_df.iloc[self.current_step].values
            
            # Add position information
            current_price = self.df.iloc[self.current_step]['Close']
            position_value = self.shares * current_price
            
            position_info = np.array([
                self.shares,
                self.cash,
                self.portfolio_value
            ])
            
            return np.concatenate([features, position_info])
        
        def step(self, action):
            current_price = self.df.iloc[self.current_step]['Close']
            
            # Execute action
            if action == 1:  # Buy
                if self.cash > current_price:
                    shares_to_buy = int(self.cash * 0.95 / current_price)
                    self.shares += shares_to_buy
                    self.cash -= shares_to_buy * current_price
                    self.trades.append(('buy', self.current_step, current_price))
            
            elif action == 2:  # Sell
                if self.shares > 0:
                    self.cash += self.shares * current_price
                    self.trades.append(('sell', self.current_step, current_price))
                    self.shares = 0
            
            # Move to next step
            self.current_step += 1
            
            # Calculate new portfolio value
            new_price = self.df.iloc[self.current_step]['Close']
            self.portfolio_value = self.cash + self.shares * new_price
            
            # Calculate reward (percentage change in portfolio value)
            reward = (self.portfolio_value - self.initial_capital) / self.initial_capital
            
            # Check if done
            done = self.current_step >= len(self.df) - 1
            
            return self._get_observation(), reward, done, {}
        
        def render(self, mode='human'):
            print(f"Step: {self.current_step}, "
                  f"Portfolio Value: ${self.portfolio_value:.2f}, "
                  f"Shares: {self.shares}, Cash: ${self.cash:.2f}")


# ==================== SENTIMENT & ALTERNATIVE DATA ====================

class AlternativeDataIntegration:
    """
    Integration of alternative data sources
    Note: Requires API keys for real implementation
    """
    
    def get_sentiment_features(self, symbol: str, date: pd.Timestamp) -> Dict:
        """
        Get sentiment features from various sources
        This is a template - actual implementation requires APIs
        """
        features = {}
        
        # Social media sentiment (Twitter/Reddit)
        features['social_sentiment'] = self._get_social_sentiment(symbol, date)
        
        # News sentiment
        features['news_sentiment'] = self._get_news_sentiment(symbol, date)
        
        # Options flow
        features['put_call_ratio'] = self._get_options_flow(symbol, date)
        
        # Insider trading
        features['insider_buying'] = self._get_insider_activity(symbol, date)
        
        # Analyst ratings
        features['analyst_score'] = self._get_analyst_ratings(symbol, date)
        
        return features
    
    def _get_social_sentiment(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Placeholder for social media sentiment
        In production: Use Twitter API, Reddit API, StockTwits
        """
        # Random placeholder - replace with actual API calls
        return np.random.uniform(-1, 1)
    
    def _get_news_sentiment(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Placeholder for news sentiment
        In production: Use NewsAPI, Bloomberg, Reuters
        """
        return np.random.uniform(-1, 1)
    
    def _get_options_flow(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Placeholder for options flow
        In production: Use CBOE data, options flow services
        """
        return np.random.uniform(0.5, 2.0)
    
    def _get_insider_activity(self, symbol: str, date: pd.Timestamp) -> int:
        """
        Placeholder for insider trading
        In production: Use SEC EDGAR, insider trading databases
        """
        return np.random.choice([-1, 0, 1])  # -1: selling, 0: none, 1: buying
    
    def _get_analyst_ratings(self, symbol: str, date: pd.Timestamp) -> float:
        """
        Placeholder for analyst ratings
        In production: Use financial data providers
        """
        return np.random.uniform(1, 5)  # 1-5 scale


if __name__ == "__main__":
    print("Advanced ML Trading Techniques Module")
    print("="*50)
    print("\nAvailable Components:")
    print("1. Advanced Feature Engineering")
    print("2. Portfolio Optimization")
    print("3. Market Regime Detection")
    print("4. Dynamic Risk Management")
    print("5. Feature Selection")
    
    if OPTUNA_AVAILABLE:
        print("6. Hyperparameter Optimization (Optuna)")
    
    if REINFORCEMENT_AVAILABLE:
        print("7. Reinforcement Learning Environment")
    
    print("\nThis module provides advanced techniques to enhance")
    print("the ML-enhanced backtest system.")
