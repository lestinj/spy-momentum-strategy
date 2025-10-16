"""
Improved data fetcher with SPY options and updated macro indicators
"""
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
from typing import Optional, Tuple
import numpy as np


class DataFetcher:
    """Fetch SPY price, options, and macro indicator data"""
    
    def __init__(self):
        self.spy = yf.Ticker('SPY')
    
    def fetch_spy_price(self, period='60d', interval='1h') -> pd.DataFrame:
        """Fetch SPY price data with proper error handling"""
        try:
            df = self.spy.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No price data returned for period={period}, interval={interval}")
            return df
        except Exception as e:
            print(f"Error fetching SPY price data: {e}")
            return pd.DataFrame()
    
    def fetch_spy_options(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch SPY options chain - calls and puts
        Returns: (calls_df, puts_df)
        """
        try:
            # Get available expiration dates
            expirations = self.spy.options
            if not expirations:
                raise ValueError("No options expiration dates available")
            
            # Get options for nearest expiration
            nearest_exp = expirations[0]
            opt_chain = self.spy.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Add useful derived columns
            calls['midPrice'] = (calls['bid'] + calls['ask']) / 2
            puts['midPrice'] = (puts['bid'] + puts['ask']) / 2
            
            # Calculate moneyness (strike / spot)
            current_price = self.fetch_spy_price(period='1d', interval='1d')['Close'].iloc[-1]
            calls['moneyness'] = calls['strike'] / current_price
            puts['moneyness'] = puts['strike'] / current_price
            
            return calls, puts
            
        except Exception as e:
            print(f"Error fetching SPY options: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def fetch_options_signals(self) -> dict:
        """
        Generate trading signals from options data:
        - Put/Call ratio
        - Implied volatility skew
        - Open interest analysis
        """
        calls, puts = self.fetch_spy_options()
        
        if calls.empty or puts.empty:
            return {}
        
        signals = {}
        
        # Put/Call Ratio (volume and open interest)
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        signals['pc_ratio_volume'] = total_put_volume / (total_call_volume + 1e-9)
        
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        signals['pc_ratio_oi'] = total_put_oi / (total_call_oi + 1e-9)
        
        # IV skew (OTM put IV vs OTM call IV)
        otm_calls = calls[calls['moneyness'] > 1.01]  # Calls above current price
        otm_puts = puts[puts['moneyness'] < 0.99]     # Puts below current price
        
        if not otm_calls.empty and not otm_puts.empty:
            avg_call_iv = otm_calls['impliedVolatility'].mean()
            avg_put_iv = otm_puts['impliedVolatility'].mean()
            signals['iv_skew'] = avg_put_iv - avg_call_iv
        else:
            signals['iv_skew'] = 0
        
        # Max pain (strike with most open interest)
        calls['total_oi'] = calls['openInterest']
        puts['total_oi'] = puts['openInterest']
        combined_oi = pd.concat([
            calls[['strike', 'total_oi']].rename(columns={'total_oi': 'call_oi'}),
            puts[['strike', 'total_oi']].rename(columns={'total_oi': 'put_oi'})
        ]).groupby('strike').sum().fillna(0)
        
        combined_oi['total'] = combined_oi['call_oi'] + combined_oi['put_oi']
        signals['max_pain'] = combined_oi['total'].idxmax()
        
        # Current SPY price
        signals['current_price'] = self.fetch_spy_price(period='1d', interval='1d')['Close'].iloc[-1]
        
        # Bullish/bearish signal from PC ratio
        # PC ratio < 0.7 = bullish, > 1.0 = bearish
        if signals['pc_ratio_volume'] < 0.7:
            signals['options_sentiment'] = 'bullish'
        elif signals['pc_ratio_volume'] > 1.0:
            signals['options_sentiment'] = 'bearish'
        else:
            signals['options_sentiment'] = 'neutral'
        
        return signals
    
    def fetch_macro_indicators(self, lookback_days=365) -> pd.DataFrame:
        """
        Fetch updated macro indicators from FRED
        Using indicators that are currently updated
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        indicators = {}
        
        # Dictionary of FRED series that are CURRENTLY UPDATED
        fred_series = {
            'UMCSENT': 'consumer_sentiment',      # U Michigan Consumer Sentiment
            'UNRATE': 'unemployment_rate',        # Unemployment Rate
            'DGS10': 'treasury_10y',              # 10-Year Treasury Rate
            'DGS2': 'treasury_2y',                # 2-Year Treasury Rate
            'T10Y2Y': 'yield_curve',              # 10Y-2Y Yield Spread
            'DCOILWTICO': 'oil_price',            # Oil Price (WTI)
            'VIXCLS': 'vix',                      # VIX
            'DEXUSEU': 'usd_eur',                 # USD/EUR exchange rate
            'INDPRO': 'industrial_production',    # Industrial Production Index
            'PAYEMS': 'nonfarm_payroll',          # Nonfarm Payrolls
        }
        
        for fred_code, name in fred_series.items():
            try:
                data = pdr.DataReader(fred_code, 'fred', start_date, end_date)
                indicators[name] = data.iloc[:, 0]
                print(f"✓ Fetched {name} ({fred_code})")
            except Exception as e:
                print(f"✗ Failed to fetch {name} ({fred_code}): {e}")
                continue
        
        if not indicators:
            print("WARNING: No macro indicators fetched successfully")
            return pd.DataFrame()
        
        # Combine all indicators
        df = pd.DataFrame(indicators)
        
        # Forward fill missing data (macro data is often monthly/weekly)
        df = df.ffill()
        
        # Calculate momentum signals
        df['consumer_bullish'] = df['consumer_sentiment'] > df['consumer_sentiment'].rolling(60).mean()
        df['unemployment_improving'] = df['unemployment_rate'] < df['unemployment_rate'].rolling(60).mean()
        df['yield_curve_positive'] = df['yield_curve'] > 0  # Not inverted
        df['oil_stable'] = df['oil_price'].pct_change(20).abs() < 0.15  # Less than 15% change
        df['vix_low'] = df['vix'] < 20  # VIX below 20 is calm
        df['industrial_growth'] = df['industrial_production'] > df['industrial_production'].shift(3)
        
        # Count positive signals
        signal_cols = ['consumer_bullish', 'unemployment_improving', 'yield_curve_positive', 
                      'oil_stable', 'vix_low', 'industrial_growth']
        df['positive_macro_count'] = df[signal_cols].sum(axis=1)
        
        # Overall macro trend (bullish if > 3 positive signals)
        df['macro_bullish'] = df['positive_macro_count'] >= 3
        
        return df
    
    def get_latest_macro_signal(self) -> dict:
        """Get the most recent macro signal"""
        df = self.fetch_macro_indicators(lookback_days=90)
        
        if df.empty:
            return {'macro_bullish': True, 'positive_count': 0}  # Default permissive
        
        latest = df.iloc[-1]
        
        return {
            'macro_bullish': bool(latest['macro_bullish']),
            'positive_count': int(latest['positive_macro_count']),
            'consumer_sentiment': latest.get('consumer_sentiment', None),
            'unemployment': latest.get('unemployment_rate', None),
            'yield_curve': latest.get('yield_curve', None),
            'vix': latest.get('vix', None),
        }


# Convenience functions for backward compatibility
def fetch_spy_price_data(period='60d', interval='1h'):
    """Backward compatible function"""
    fetcher = DataFetcher()
    return fetcher.fetch_spy_price(period, interval)


def fetch_macro_indicator(indicator_name: str, start, end):
    """
    Backward compatible function - but now properly handles updated indicators
    """
    fetcher = DataFetcher()
    df = fetcher.fetch_macro_indicators(lookback_days=(end - start).days)
    
    # Return the first available indicator if specific one requested
    if not df.empty:
        return df[['positive_macro_count']].rename(columns={'positive_macro_count': indicator_name})
    else:
        return pd.DataFrame()
