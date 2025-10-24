"""
V4.8 MONTE CARLO - QUICK START RUNNER
═══════════════════════════════════════════════════════════════════════════════

This script makes it super easy to run Monte Carlo simulations on your v4.8 trades.

USAGE:
    python run_monte_carlo.py

REQUIREMENTS:
    1. Have your v4.8 trades CSV file
    2. Know your starting capital
    3. Run this script!

The script will:
    ✓ Load your historical trades
    ✓ Run 10,000 Monte Carlo simulations
    ✓ Generate comprehensive charts
    ✓ Save detailed statistics
    ✓ Show you what to expect going forward
"""

import os
import sys
from v48_monte_carlo_simulator import run_monte_carlo_simulation

def find_trades_csv():
    """Try to automatically find v4.8 trades CSV"""
    
    possible_names = [
        'v48_stock_trades.csv',
        'v48_trades.csv',
        'v48_stock_trades_*.csv',
        'v48_true_original_trades.csv',
        'v48_6strategies_trades.csv',
    ]
    
    for pattern in possible_names:
        if '*' in pattern:
            import glob
            files = glob.glob(pattern)
            if files:
                return files[0]
        else:
            if os.path.exists(pattern):
                return pattern
    
    return None

def main():
    """Main execution function"""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*20 + "V4.8 MONTE CARLO - QUICK START" + " "*28 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # =========================================================================
    # STEP 1: Find trades CSV
    # =========================================================================
    print("STEP 1: Locating trades CSV file...")
    print("-" * 80)
    
    trades_csv = find_trades_csv()
    
    if trades_csv:
        print(f"✓ Found: {trades_csv}")
        use_this = input(f"\nUse this file? (Y/n): ").strip().lower()
        if use_this in ['n', 'no']:
            trades_csv = None
    
    if not trades_csv:
        print("\n❌ Could not auto-detect trades CSV.")
        trades_csv = input("Enter path to your trades CSV: ").strip()
        
        if not os.path.exists(trades_csv):
            print(f"\n❌ Error: File not found: {trades_csv}")
            print("\nMake sure you've run your v4.8 backtest first!")
            return
    
    print(f"\n✓ Using trades file: {trades_csv}\n")
    
    # =========================================================================
    # STEP 2: Get configuration
    # =========================================================================
    print("\nSTEP 2: Configuration")
    print("-" * 80)
    
    # Starting capital
    default_capital = 10000
    capital_input = input(f"Starting capital (default ${default_capital:,}): ").strip()
    initial_capital = int(capital_input) if capital_input else default_capital
    
    # Number of simulations
    default_sims = 10000
    sims_input = input(f"Number of simulations (default {default_sims:,}): ").strip()
    simulations = int(sims_input) if sims_input else default_sims
    
    # Trading days forward
    print("\nHow far forward to simulate?")
    print("  1. 3 months (63 days)")
    print("  2. 6 months (126 days)")
    print("  3. 1 year (252 days) - RECOMMENDED")
    print("  4. 2 years (504 days)")
    print("  5. 3 years (756 days)")
    
    days_choice = input("\nChoice (1-5, default 3): ").strip()
    days_map = {
        '1': 63,
        '2': 126,
        '3': 252,
        '4': 504,
        '5': 756,
    }
    trading_days = days_map.get(days_choice, 252)
    
    # Trades per day
    default_tpd = 1.5
    tpd_input = input(f"Avg trades per day (default {default_tpd}): ").strip()
    trades_per_day = float(tpd_input) if tpd_input else default_tpd
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Trades CSV:        {trades_csv}")
    print(f"Initial Capital:   ${initial_capital:,}")
    print(f"Simulations:       {simulations:,}")
    print(f"Trading Days:      {trading_days} ({trading_days/252:.1f} years)")
    print(f"Trades/Day:        {trades_per_day}")
    print("="*80)
    
    confirm = input("\nProceed with simulation? (Y/n): ").strip().lower()
    if confirm in ['n', 'no']:
        print("\n❌ Cancelled.")
        return
    
    # =========================================================================
    # STEP 3: Run simulation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Running Monte Carlo Simulation")
    print("="*80)
    print("\n⏳ This may take a few minutes depending on number of simulations...")
    print("    (Grab a coffee ☕)\n")
    
    simulator = run_monte_carlo_simulation(
        trades_csv_path=trades_csv,
        initial_capital=initial_capital,
        simulations=simulations,
        trading_days=trading_days,
        trades_per_day=trades_per_day
    )
    
    if simulator:
        # =====================================================================
        # STEP 4: Quick interpretation
        # =====================================================================
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + " "*25 + "QUICK INTERPRETATION" + " "*33 + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝\n")
        
        stats = simulator.stats
        
        print("🎯 EXPECTED OUTCOME (Most Likely):")
        print(f"   Final Capital: ${stats['final_capital_median']:,.0f}")
        print(f"   Total Return:  {stats['return_median']:.1f}%")
        print(f"   Annualized:    {stats['cagr_median']:.1f}%")
        
        print("\n📊 PROBABILITY ANALYSIS:")
        if stats['prob_profit'] >= 85:
            print(f"   ✅ {stats['prob_profit']:.1f}% chance of profit - EXCELLENT")
        elif stats['prob_profit'] >= 70:
            print(f"   ⚠️  {stats['prob_profit']:.1f}% chance of profit - GOOD")
        else:
            print(f"   ❌ {stats['prob_profit']:.1f}% chance of profit - RISKY")
        
        if stats['prob_double'] >= 40:
            print(f"   ✅ {stats['prob_double']:.1f}% chance of doubling - HIGH")
        elif stats['prob_double'] >= 20:
            print(f"   📈 {stats['prob_double']:.1f}% chance of doubling - MODERATE")
        else:
            print(f"   📉 {stats['prob_double']:.1f}% chance of doubling - LOW")
        
        print("\n⚠️  RISK ANALYSIS:")
        if abs(stats['max_dd_median']) <= 30:
            print(f"   ✅ Median drawdown: {stats['max_dd_median']:.1f}% - MANAGEABLE")
        elif abs(stats['max_dd_median']) <= 50:
            print(f"   ⚠️  Median drawdown: {stats['max_dd_median']:.1f}% - MODERATE")
        else:
            print(f"   ❌ Median drawdown: {stats['max_dd_median']:.1f}% - HIGH")
        
        print(f"   🔴 Worst case DD:   {stats['max_dd_worst']:.1f}%")
        
        if stats['prob_ruin'] <= 0.5:
            print(f"   ✅ Risk of ruin:    {stats['prob_ruin']:.2f}% - MINIMAL")
        elif stats['prob_ruin'] <= 2:
            print(f"   ⚠️  Risk of ruin:    {stats['prob_ruin']:.2f}% - LOW")
        else:
            print(f"   ❌ Risk of ruin:    {stats['prob_ruin']:.2f}% - CONCERNING")
        
        print("\n💰 RECOMMENDATION:")
        
        # Decision logic
        if (stats['prob_profit'] >= 85 and 
            abs(stats['max_dd_median']) <= 50 and 
            stats['prob_ruin'] <= 1):
            print("   ✅ STRONG STRATEGY - Proceed with confidence!")
            print("      • Start with 25-50% of capital")
            print("      • Scale up gradually")
            print("      • Monitor monthly")
        
        elif (stats['prob_profit'] >= 70 and 
              abs(stats['max_dd_median']) <= 60 and 
              stats['prob_ruin'] <= 2):
            print("   ⚠️  MODERATE STRATEGY - Proceed with caution")
            print("      • Start with 10-25% of capital")
            print("      • Reduce position sizing by 50%")
            print("      • Very close monitoring")
        
        else:
            print("   ❌ RISKY STRATEGY - Consider improvements before live trading")
            print("      • Paper trade first")
            print("      • Improve win rate or reduce position sizing")
            print("      • Re-run simulation after improvements")
        
        print("\n" + "="*80)
        print("✓ Check the 'trading_results' folder for detailed charts and CSVs")
        print("="*80)
        
    else:
        print("\n❌ Simulation failed. Check your trades CSV and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
