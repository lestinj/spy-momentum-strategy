#!/usr/bin/env python3
"""
BACKTEST COMPARISON RUNNER
==========================
Easy testing of ML-Optimized V49 across different periods
"""

import subprocess
import sys
from datetime import datetime

def run_backtest(start_date, capital=100000):
    """Run the ML-optimized backtest and extract results"""
    print(f"\n{'='*60}")
    print(f"Testing from {start_date} to present with ${capital:,}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, 'ml_optimized_v49.py', start_date, str(capital)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout
        
        # Extract metrics
        metrics = {}
        
        # Find CAGR
        for line in output.split('\n'):
            if 'CAGR:' in line and '%' in line:
                try:
                    cagr = float(line.split(':')[1].split('%')[0].strip())
                    metrics['cagr'] = cagr
                except:
                    pass
            elif 'Sharpe Ratio:' in line:
                try:
                    sharpe = float(line.split(':')[1].strip())
                    metrics['sharpe'] = sharpe
                except:
                    pass
            elif 'Max Drawdown:' in line and '%' in line:
                try:
                    dd = float(line.split(':')[1].split('%')[0].strip())
                    metrics['max_dd'] = dd
                except:
                    pass
            elif 'Win Rate:' in line and '%' in line:
                try:
                    wr = float(line.split(':')[1].split('%')[0].strip())
                    metrics['win_rate'] = wr
                except:
                    pass
            elif 'Final Equity:' in line:
                try:
                    equity_str = line.split('$')[1].replace(',', '')
                    equity = float(equity_str)
                    metrics['final_equity'] = equity
                except:
                    pass
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("Timeout - backtest took too long")
        return None
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("ML-OPTIMIZED V49 - MULTI-PERIOD TESTING")
    print("Strategy: Take ALL V49 signals, size by ML confidence")
    print("="*80)
    
    # Test periods
    test_periods = [
        ('2015-01-01', '10 years'),
        ('2016-01-01', '9 years'),
        ('2017-01-01', '8 years'),
        ('2018-01-01', '7 years'),
        ('2019-01-01', '6 years'),
        ('2020-01-01', '5 years'),
        ('2021-01-01', '4 years'),
        ('2022-01-01', '3 years'),
        ('2023-01-01', '2 years'),
        ('2024-01-01', '1.8 years'),
    ]
    
    capital = 100000
    results = []
    
    print(f"\nInitial Capital: ${capital:,}")
    print(f"Testing each period independently...")
    
    # Summary table header
    print("\n" + "="*90)
    print(f"{'Period':<15} {'Duration':<10} {'CAGR':<12} {'Sharpe':<10} {'Max DD':<12} {'Win Rate':<12} {'Final Equity':<15}")
    print("="*90)
    
    for start_date, duration in test_periods:
        metrics = run_backtest(start_date, capital)
        
        if metrics and 'cagr' in metrics:
            print(f"{start_date:<15} {duration:<10} "
                  f"{metrics['cagr']:>8.1f}%    "
                  f"{metrics.get('sharpe', 0):>7.2f}    "
                  f"{metrics.get('max_dd', 0):>8.1f}%    "
                  f"{metrics.get('win_rate', 0):>8.1f}%    "
                  f"${metrics.get('final_equity', 0):>12,.0f}")
            
            results.append({
                'period': start_date,
                'duration': duration,
                **metrics
            })
        else:
            print(f"{start_date:<15} {duration:<10} Failed to extract metrics")
    
    # Calculate averages
    if results:
        print("\n" + "="*90)
        print("SUMMARY STATISTICS")
        print("="*90)
        
        avg_cagr = sum(r['cagr'] for r in results) / len(results)
        avg_sharpe = sum(r.get('sharpe', 0) for r in results) / len(results)
        avg_dd = sum(r.get('max_dd', 0) for r in results) / len(results)
        avg_wr = sum(r.get('win_rate', 0) for r in results) / len(results)
        
        print(f"\n📊 AVERAGE PERFORMANCE:")
        print(f"   Average CAGR:        {avg_cagr:>8.1f}%")
        print(f"   Average Sharpe:      {avg_sharpe:>8.2f}")
        print(f"   Average Max DD:      {avg_dd:>8.1f}%")
        print(f"   Average Win Rate:    {avg_wr:>8.1f}%")
        
        # Best and worst
        best_cagr = max(results, key=lambda x: x['cagr'])
        worst_cagr = min(results, key=lambda x: x['cagr'])
        
        print(f"\n📈 BEST PERFORMANCE:")
        print(f"   Period: {best_cagr['period']} ({best_cagr['cagr']:.1f}% CAGR)")
        
        print(f"\n📉 WORST PERFORMANCE:")
        print(f"   Period: {worst_cagr['period']} ({worst_cagr['cagr']:.1f}% CAGR)")
        
        # Success rate
        above_target = sum(1 for r in results if r['cagr'] > 77)
        print(f"\n🎯 SUCCESS RATE:")
        print(f"   Periods above 77% CAGR: {above_target}/{len(results)}")
        print(f"   Periods above 100% CAGR: {sum(1 for r in results if r['cagr'] > 100)}/{len(results)}")
        
        # Save results
        import json
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to test_results.json")
    
    print("\n" + "="*90)
    print("✅ Testing complete!")
    print("="*90)

if __name__ == "__main__":
    main()
