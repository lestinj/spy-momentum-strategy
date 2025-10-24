#!/usr/bin/env python3
"""
BACKTEST CAGR COMPARISON: V49 vs ML-Enhanced
============================================
Compares performance from various start years to October 2025
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime

# Import both backtest classes
from v49_backtest import V49Backtest
from ml_enhanced_backtest import MLEnhancedV49Proper

def run_comparison(start_year, initial_capital=100000):
    """Run both backtests from a given year and extract key metrics"""
    
    start_date = f'{start_year}-01-01'
    results = {}
    
    # Run V49 Backtest
    print(f"\nRunning V49 from {start_year}...", end='', flush=True)
    v49 = V49Backtest(initial_capital=initial_capital)
    
    if v49.load_data(start_date=start_date):
        # Suppress output
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        v49.run_backtest()
        
        sys.stdout = old_stdout
        
        if v49.equity_curve:
            equity_df = pd.DataFrame(v49.equity_curve)
            final_equity = equity_df['equity'].iloc[-1]
            
            # Calculate CAGR
            start = equity_df['date'].iloc[0]
            end = equity_df['date'].iloc[-1]
            years = (end - start).days / 365.25
            
            if years > 0:
                cagr = (((final_equity / initial_capital) ** (1/years)) - 1) * 100
                
                # Sharpe ratio
                equity_df['returns'] = equity_df['equity'].pct_change()
                sharpe = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std() if equity_df['returns'].std() > 0 else 0
                
                # Max drawdown
                equity_df['cummax'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
                max_dd = equity_df['drawdown'].min()
                
                # Win rate
                if v49.trades:
                    trades_df = pd.DataFrame(v49.trades)
                    win_rate = (len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100) if len(trades_df) > 0 else 0
                else:
                    win_rate = 0
                
                results['v49'] = {
                    'cagr': cagr,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'win_rate': win_rate,
                    'final_equity': final_equity,
                    'years': years
                }
                print(f" ✓ CAGR: {cagr:.1f}%")
    
    # Run ML Enhanced Backtest
    print(f"Running ML from {start_year}...", end='', flush=True)
    ml = MLEnhancedV49Proper(initial_capital=initial_capital)
    
    if ml.load_data(start_date=start_date):
        # Suppress output
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        ml.run_backtest()
        
        sys.stdout = old_stdout
        
        if ml.equity_curve:
            equity_df = pd.DataFrame(ml.equity_curve)
            final_equity = equity_df['equity'].iloc[-1]
            
            # Calculate CAGR
            start = equity_df['date'].iloc[0]
            end = equity_df['date'].iloc[-1]
            years = (end - start).days / 365.25
            
            if years > 0:
                cagr = (((final_equity / initial_capital) ** (1/years)) - 1) * 100
                
                # Sharpe ratio
                equity_df['returns'] = equity_df['equity'].pct_change()
                sharpe = np.sqrt(252) * equity_df['returns'].mean() / equity_df['returns'].std() if equity_df['returns'].std() > 0 else 0
                
                # Max drawdown
                equity_df['cummax'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
                max_dd = equity_df['drawdown'].min()
                
                # Win rate
                if ml.trades:
                    trades_df = pd.DataFrame(ml.trades)
                    win_rate = (len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100) if len(trades_df) > 0 else 0
                else:
                    win_rate = 0
                
                results['ml'] = {
                    'cagr': cagr,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'win_rate': win_rate,
                    'final_equity': final_equity,
                    'years': years
                }
                print(f" ✓ CAGR: {cagr:.1f}%")
    
    return results

def main():
    print("\n" + "="*100)
    print("CUMULATIVE BACKTEST COMPARISON: V49 vs ML-Enhanced")
    print("Testing from various years to October 2025")
    print("="*100)
    
    # Years to test (going back to when sufficient data is available)
    test_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    initial_capital = 100000
    
    print(f"\nInitial Capital: ${initial_capital:,}")
    print(f"End Date: October 2025")
    
    all_results = []
    
    for year in test_years:
        results = run_comparison(year, initial_capital)
        
        if 'v49' in results and 'ml' in results:
            v49 = results['v49']
            ml = results['ml']
            
            all_results.append({
                'start_year': year,
                'period': f"{year}-2025",
                'years': v49['years'],
                'v49_cagr': v49['cagr'],
                'ml_cagr': ml['cagr'],
                'difference': ml['cagr'] - v49['cagr'],
                'v49_sharpe': v49['sharpe'],
                'ml_sharpe': ml['sharpe'],
                'v49_max_dd': v49['max_dd'],
                'ml_max_dd': ml['max_dd'],
                'v49_win_rate': v49['win_rate'],
                'ml_win_rate': ml['win_rate'],
                'v49_final': v49['final_equity'],
                'ml_final': ml['final_equity']
            })
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Display results table
        print("\n" + "="*100)
        print("CAGR COMPARISON TABLE")
        print("="*100)
        print(f"\n{'Period':<15} {'Years':<8} {'V49 CAGR':<12} {'ML CAGR':<12} {'Difference':<12} {'Winner':<10}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            winner = "ML" if row['difference'] > 0 else "V49" if row['difference'] < 0 else "TIE"
            print(f"{row['period']:<15} {row['years']:<8.1f} {row['v49_cagr']:<12.1f} {row['ml_cagr']:<12.1f} "
                  f"{row['difference']:+12.1f} {winner:<10}")
        
        # Summary statistics
        print("\n" + "="*100)
        print("PERFORMANCE SUMMARY")
        print("="*100)
        
        print(f"\n📊 AVERAGE METRICS:")
        print(f"   V49 Average CAGR:     {df['v49_cagr'].mean():>8.1f}%")
        print(f"   ML Average CAGR:      {df['ml_cagr'].mean():>8.1f}%")
        print(f"   Difference:           {df['difference'].mean():>+8.1f}%")
        print(f"\n   V49 Avg Sharpe:       {df['v49_sharpe'].mean():>8.2f}")
        print(f"   ML Avg Sharpe:        {df['ml_sharpe'].mean():>8.2f}")
        print(f"\n   V49 Avg Max DD:       {df['v49_max_dd'].mean():>8.1f}%")
        print(f"   ML Avg Max DD:        {df['ml_max_dd'].mean():>8.1f}%")
        print(f"\n   V49 Avg Win Rate:     {df['v49_win_rate'].mean():>8.1f}%")
        print(f"   ML Avg Win Rate:      {df['ml_win_rate'].mean():>8.1f}%")
        
        # Best performances
        print(f"\n📈 BEST PERFORMANCES:")
        best_v49 = df.loc[df['v49_cagr'].idxmax()]
        best_ml = df.loc[df['ml_cagr'].idxmax()]
        print(f"   Best V49:  {best_v49['period']} ({best_v49['v49_cagr']:.1f}% CAGR)")
        print(f"   Best ML:   {best_ml['period']} ({best_ml['ml_cagr']:.1f}% CAGR)")
        
        # Win count
        ml_wins = (df['difference'] > 0).sum()
        v49_wins = (df['difference'] < 0).sum()
        ties = (df['difference'] == 0).sum()
        
        print(f"\n🏆 HEAD-TO-HEAD RESULTS:")
        print(f"   ML Won:    {ml_wins}/{len(df)} periods")
        print(f"   V49 Won:   {v49_wins}/{len(df)} periods")
        if ties > 0:
            print(f"   Tied:      {ties}/{len(df)} periods")
        
        # Overall winner
        print(f"\n" + "="*100)
        avg_diff = df['difference'].mean()
        if avg_diff > 0:
            print(f"🎯 OVERALL WINNER: ML-Enhanced Strategy")
            print(f"   Average outperformance: {avg_diff:+.1f}% CAGR")
        elif avg_diff < 0:
            print(f"🎯 OVERALL WINNER: V49 Base Strategy")
            print(f"   Average outperformance: {-avg_diff:+.1f}% CAGR")
        else:
            print(f"🎯 RESULT: Strategies performed equally")
        print("="*100)
        
        # Save to CSV
        df.to_csv('cagr_comparison_results.csv', index=False)
        print(f"\n📁 Results saved to: cagr_comparison_results.csv")

if __name__ == "__main__":
    main()
    print("\n✅ Comparison complete!")
