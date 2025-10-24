"""
Symbol Screening Tool - Find Best Performers for Strategy
Tests multiple candidates to replace underperformers like ROKU
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SymbolScreener:
    """Screen symbols for compatibility with momentum + mean reversion strategies"""
    
    def __init__(self, start_date: str = "2020-01-01", end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Candidate symbols to test
        self.tech_candidates = [
            # Current portfolio
            "NVDA", "AMZN", "ORCL", "MSFT",
            
            # ROKU alternatives (streaming/content)
            "NFLX", "DIS", "PARA",
            
            # Semiconductor (similar to NVDA)
            "AMD", "AVGO", "TSM", "QCOM", "MU",
            
            # Mega-cap tech (stable)
            "GOOGL", "META", "AAPL",
            
            # Software/Cloud (similar profile)
            "CRM", "ADBE", "NOW", "SNOW", "DDOG",
            
            # Growth tech
            "TSLA", "SQ", "SHOP", "PLTR",
            
            # Enterprise
            "IBM", "CSCO", "INTC"
        ]
    
    def get_basic_metrics(self, symbol: str) -> dict:
        """Get basic performance and volatility metrics"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=self.start_date, end=self.end_date, interval='1d')
            
            if df.empty or len(df) < 100:
                return None
            
            df.columns = [c.lower() for c in df.columns]
            
            # Calculate metrics
            total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
            
            # Volatility
            daily_returns = df['close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Sharpe (rough)
            mean_return = daily_returns.mean() * 252
            sharpe = mean_return / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            
            # Drawdown
            rolling_max = df['close'].expanding().max()
            drawdown = ((df['close'] - rolling_max) / rolling_max * 100).min()
            
            # Trend strength
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            days_above_50 = (df['close'] > df['sma_50']).sum() / len(df) * 100
            days_above_200 = (df['close'] > df['sma_200']).sum() / len(df) * 100
            
            # Mean reversion opportunities (approximate)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain / loss))
            
            oversold_days = (rsi < 30).sum()
            overbought_days = (rsi > 70).sum()
            
            return {
                'symbol': symbol,
                'total_return': total_return,
                'annualized_return': ((1 + total_return/100) ** (365.25 / len(df)) - 1) * 100,
                'volatility': volatility,
                'sharpe': sharpe,
                'max_drawdown': drawdown,
                'days_above_50ma': days_above_50,
                'days_above_200ma': days_above_200,
                'oversold_days': oversold_days,
                'overbought_days': overbought_days,
                'mean_reversion_score': oversold_days + overbought_days,
                'trading_days': len(df)
            }
            
        except Exception as e:
            print(f"Error with {symbol}: {e}")
            return None
    
    def screen_all(self) -> pd.DataFrame:
        """Screen all candidate symbols"""
        print(f"\n{'='*80}")
        print(f"SCREENING {len(self.tech_candidates)} CANDIDATE SYMBOLS")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"{'='*80}\n")
        
        results = []
        
        for symbol in self.tech_candidates:
            print(f"Analyzing {symbol}...", end=" ")
            metrics = self.get_basic_metrics(symbol)
            
            if metrics:
                results.append(metrics)
                print(f"✓ Return: {metrics['total_return']:.1f}%")
            else:
                print("✗ Failed")
        
        df = pd.DataFrame(results)
        
        # Calculate composite score
        df['momentum_score'] = (
            df['days_above_50ma'] * 0.3 +
            df['days_above_200ma'] * 0.2 +
            df['sharpe'] * 20
        )
        
        df['stability_score'] = (
            100 - abs(df['max_drawdown']) * 0.5 +
            (50 - abs(df['volatility'] - 40)) * 0.5
        )
        
        df['composite_score'] = (
            df['annualized_return'] * 0.3 +
            df['momentum_score'] * 0.3 +
            df['stability_score'] * 0.2 +
            df['mean_reversion_score'] * 0.2
        )
        
        return df
    
    def display_results(self, df: pd.DataFrame):
        """Display screening results with recommendations"""
        print(f"\n{'='*80}")
        print("SCREENING RESULTS - SORTED BY COMPOSITE SCORE")
        print(f"{'='*80}\n")
        
        # Sort by composite score
        df_sorted = df.sort_values('composite_score', ascending=False)
        
        print("TOP 10 BEST CANDIDATES:")
        print("-" * 100)
        print(f"{'Rank':<6} {'Symbol':<8} {'Return%':<10} {'Sharpe':<8} {'MaxDD%':<10} "
              f"{'Vol%':<8} {'MR Score':<10} {'Composite':<10}")
        print("-" * 100)
        
        for i, row in enumerate(df_sorted.head(10).itertuples(), 1):
            print(f"{i:<6} {row.symbol:<8} {row.total_return:<10.1f} {row.sharpe:<8.2f} "
                  f"{row.max_drawdown:<10.1f} {row.volatility:<8.1f} "
                  f"{row.mean_reversion_score:<10.0f} {row.composite_score:<10.1f}")
        
        print("\n" + "="*80)
        print("MEAN REVERSION POTENTIAL (Most Oversold/Overbought Days):")
        print("="*80)
        df_mr = df.sort_values('mean_reversion_score', ascending=False)
        print(f"\n{'Symbol':<10} {'Oversold Days':<15} {'Overbought Days':<15} {'Total MR Days':<15}")
        print("-" * 60)
        for row in df_mr.head(10).itertuples():
            print(f"{row.symbol:<10} {row.oversold_days:<15} {row.overbought_days:<15} "
                  f"{row.mean_reversion_score:<15}")
        
        print("\n" + "="*80)
        print("BOTTOM 5 WORST PERFORMERS (Avoid These):")
        print("="*80)
        df_worst = df.sort_values('composite_score', ascending=True)
        print(f"\n{'Symbol':<10} {'Return%':<10} {'Sharpe':<8} {'MaxDD%':<10} {'Why Avoid':<40}")
        print("-" * 80)
        for row in df_worst.head(5).itertuples():
            reason = []
            if row.sharpe < 0.5:
                reason.append("Low Sharpe")
            if row.max_drawdown < -60:
                reason.append("Huge DD")
            if row.total_return < 50:
                reason.append("Low Return")
            
            print(f"{row.symbol:<10} {row.total_return:<10.1f} {row.sharpe:<8.2f} "
                  f"{row.max_drawdown:<10.1f} {', '.join(reason):<40}")
        
        print("\n" + "="*80)
        print("RECOMMENDED PORTFOLIO BASED ON SCREENING:")
        print("="*80)
        
        # Recommend top 5 with diversity
        top_symbols = df_sorted.head(15)
        
        # Try to get diversity
        recommendations = []
        semiconductors = ['NVDA', 'AMD', 'AVGO', 'TSM', 'QCOM', 'MU']
        mega_cap = ['GOOGL', 'META', 'AAPL', 'MSFT', 'AMZN']
        enterprise = ['ORCL', 'CRM', 'ADBE', 'IBM', 'CSCO']
        
        # Get best from each category
        for symbol in top_symbols['symbol']:
            if len(recommendations) < 5:
                recommendations.append(symbol)
        
        print("\nSuggested 5-Symbol Portfolio:")
        for i, symbol in enumerate(recommendations[:5], 1):
            row = df_sorted[df_sorted['symbol'] == symbol].iloc[0]
            category = "Semiconductor" if symbol in semiconductors else \
                      "Mega-cap" if symbol in mega_cap else \
                      "Enterprise" if symbol in enterprise else "Growth"
            print(f"  {i}. {symbol:<8} ({category:<15}) - Return: {row['total_return']:>7.1f}%, "
                  f"Sharpe: {row['sharpe']:.2f}, MaxDD: {row['max_drawdown']:.1f}%")
        
        print("\n" + "="*80)
        print("CURRENT PORTFOLIO ANALYSIS:")
        print("="*80)
        current = ['NVDA', 'AMZN', 'ROKU', 'ORCL', 'MSFT']
        
        print(f"\n{'Symbol':<10} {'Rank':<8} {'Return%':<12} {'Sharpe':<10} {'Comments':<40}")
        print("-" * 80)
        
        for symbol in current:
            if symbol in df['symbol'].values:
                rank = df_sorted[df_sorted['symbol'] == symbol].index[0] + 1
                row = df_sorted[df_sorted['symbol'] == symbol].iloc[0]
                
                comment = "EXCELLENT" if rank <= 5 else \
                         "GOOD" if rank <= 10 else \
                         "MEDIOCRE" if rank <= 20 else \
                         "POOR - CONSIDER REPLACING"
                
                print(f"{symbol:<10} #{rank:<7} {row['total_return']:<12.1f} {row['sharpe']:<10.2f} {comment:<40}")
            else:
                print(f"{symbol:<10} {'N/A':<8} {'N/A':<12} {'N/A':<10} {'Failed to load data':<40}")
        
        print("\n" + "="*80)

def run_screening():
    """Run the symbol screening process"""
    screener = SymbolScreener(
        start_date="2020-01-01"
    )
    
    results_df = screener.screen_all()
    screener.display_results(results_df)
    
    # Save results
    results_df.to_csv('symbol_screening_results.csv', index=False)
    print("\n✓ Results saved to 'symbol_screening_results.csv'")
    
    return results_df

if __name__ == "__main__":
    df = run_screening()