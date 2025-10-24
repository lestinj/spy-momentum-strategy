"""
V4.8 MONTE CARLO - DEMO WITH SAMPLE DATA
═══════════════════════════════════════════════════════════════════════════════

This demo script generates SAMPLE trades data and runs the Monte Carlo simulator
so you can see how it works BEFORE using your real v4.8 trades.

USAGE:
    python demo_monte_carlo.py

This will:
    1. Generate 500 sample trades (similar to v4.8 characteristics)
    2. Save to sample_trades.csv
    3. Run Monte Carlo simulation
    4. Generate all charts and reports

Perfect for:
    • Testing the simulator
    • Understanding the output
    • Learning interpretation
    • Verifying installation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from v48_monte_carlo_simulator import run_monte_carlo_simulation

def generate_sample_trades(
    num_trades=500,
    win_rate=0.35,
    winner_mean_pct=45,
    winner_std_pct=30,
    loser_mean_pct=-12,
    loser_std_pct=8,
    avg_holding_days=18
):
    """
    Generate realistic sample trades similar to v4.8 characteristics
    
    Args:
        num_trades: Number of trades to generate
        win_rate: Percentage of winning trades
        winner_mean_pct: Average return % for winners
        winner_std_pct: Std dev of winner returns
        loser_mean_pct: Average return % for losers
        loser_std_pct: Std dev of loser returns
        avg_holding_days: Average days held
    """
    
    print("\n" + "="*80)
    print("GENERATING SAMPLE TRADES")
    print("="*80)
    print(f"Number of trades:      {num_trades}")
    print(f"Win rate:              {win_rate*100:.1f}%")
    print(f"Winner avg return:     {winner_mean_pct:+.1f}%")
    print(f"Loser avg return:      {loser_mean_pct:+.1f}%")
    print(f"Avg holding days:      {avg_holding_days:.1f}")
    print("="*80)
    
    trades = []
    start_date = datetime(2020, 1, 1)
    
    symbols = ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'SMCI', 'MSTR', 'CRWD']
    strategies = ['BREAKOUT', 'TREND_FOLLOW', 'MOMENTUM_ACCEL', 'PULLBACK']
    
    for i in range(num_trades):
        # Randomly determine if winner or loser
        is_winner = np.random.random() < win_rate
        
        # Generate return
        if is_winner:
            return_pct = np.random.normal(winner_mean_pct, winner_std_pct) / 100
        else:
            return_pct = np.random.normal(loser_mean_pct, loser_std_pct) / 100
        
        # Generate trade details
        entry_price = np.random.uniform(50, 500)
        shares = np.random.uniform(1, 10)
        position_size = entry_price * shares
        
        exit_price = entry_price * (1 + return_pct)
        pnl = (exit_price - entry_price) * shares
        
        # Holding period
        days_held = max(1, int(np.random.exponential(avg_holding_days)))
        
        # Dates
        entry_date = start_date + timedelta(days=i*2)
        exit_date = entry_date + timedelta(days=days_held)
        
        trade = {
            'trade_id': i + 1,
            'symbol': np.random.choice(symbols),
            'strategy': np.random.choice(strategies),
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'shares': round(shares, 2),
            'position_size': round(position_size, 2),
            'pnl': round(pnl, 2),
            'return_pct': round(return_pct * 100, 2),
            'days_held': days_held,
            'is_winner': is_winner
        }
        
        trades.append(trade)
    
    # Create DataFrame
    df = pd.DataFrame(trades)
    
    # Summary statistics
    winners = df[df['is_winner']]
    losers = df[~df['is_winner']]
    
    print(f"\n✓ Generated {len(df)} trades:")
    print(f"  Winners: {len(winners)} ({len(winners)/len(df)*100:.1f}%)")
    print(f"  Losers:  {len(losers)} ({len(losers)/len(df)*100:.1f}%)")
    print(f"  Avg winner return: {winners['return_pct'].mean():+.1f}%")
    print(f"  Avg loser return:  {losers['return_pct'].mean():+.1f}%")
    print(f"  Total P&L: ${df['pnl'].sum():,.2f}")
    
    return df

def main():
    """Run complete demo"""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " "*20 + "V4.8 MONTE CARLO SIMULATOR - DEMO" + " "*24 + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\nThis demo will:")
    print("  1. Generate 500 sample trades")
    print("  2. Save to 'sample_trades.csv'")
    print("  3. Run Monte Carlo simulation (10,000 paths)")
    print("  4. Generate all charts and reports")
    print("  5. Show you what to expect")
    
    input("\nPress ENTER to start the demo...")
    
    # =========================================================================
    # STEP 1: Generate sample trades
    # =========================================================================
    trades_df = generate_sample_trades(
        num_trades=500,
        win_rate=0.347,  # V4.8 typical win rate
        winner_mean_pct=45,
        winner_std_pct=30,
        loser_mean_pct=-12,
        loser_std_pct=8,
        avg_holding_days=18
    )
    
    # Save to CSV
    trades_df.to_csv('sample_trades.csv', index=False)
    print(f"\n✓ Saved sample trades to: sample_trades.csv")
    
    # =========================================================================
    # STEP 2: Run Monte Carlo simulation
    # =========================================================================
    print("\n" + "="*80)
    print("Running Monte Carlo Simulation...")
    print("="*80)
    print("\n⏳ This will take 1-2 minutes...")
    
    simulator = run_monte_carlo_simulation(
        trades_csv_path='sample_trades.csv',
        initial_capital=10000,
        simulations=10000,
        trading_days=252,  # 1 year
        trades_per_day=1.5
    )
    
    # =========================================================================
    # STEP 3: Interpretation
    # =========================================================================
    if simulator:
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + " "*25 + "DEMO COMPLETE!" + " "*37 + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        
        print("\n📊 What Just Happened:")
        print("="*80)
        print("1. ✓ Generated 500 realistic sample trades")
        print("2. ✓ Ran 10,000 Monte Carlo simulations")
        print("3. ✓ Calculated comprehensive statistics")
        print("4. ✓ Generated 4 visualization charts")
        print("5. ✓ Saved detailed CSVs")
        
        print("\n📁 Files Created:")
        print("-"*80)
        print("  • sample_trades.csv - The generated sample trades")
        print("  • trading_results/monte_carlo_paths_*.png")
        print("  • trading_results/distributions_*.png")
        print("  • trading_results/risk_return_*.png")
        print("  • trading_results/cumulative_prob_*.png")
        print("  • trading_results/monte_carlo_summary_*.csv")
        print("  • trading_results/monte_carlo_all_sims_*.csv")
        
        print("\n🎓 Next Steps:")
        print("-"*80)
        print("1. Open the PNG files to see the visualizations")
        print("2. Review the statistics above")
        print("3. Compare to the guide (V48_MONTE_CARLO_GUIDE.md)")
        print("4. Run your REAL v4.8 backtest")
        print("5. Use run_monte_carlo.py with your actual trades")
        
        print("\n💡 Understanding the Demo Results:")
        print("-"*80)
        
        stats = simulator.stats
        
        print("\nThese results are based on SAMPLE data with:")
        print(f"  • {trades_df['is_winner'].sum()} winners ({trades_df['is_winner'].mean()*100:.1f}%)")
        print(f"  • Avg winner: +{trades_df[trades_df['is_winner']]['return_pct'].mean():.1f}%")
        print(f"  • Avg loser:  {trades_df[~trades_df['is_winner']]['return_pct'].mean():.1f}%")
        
        print("\nYour REAL results with actual v4.8 trades will:")
        print("  • Be based on YOUR actual trade history")
        print("  • Reflect YOUR symbols and strategy")
        print("  • Show YOUR realistic forward expectations")
        
        print("\n" + "="*80)
        print("✓ Demo complete! Now try it with your real v4.8 trades!")
        print("="*80)
        
        return True
    
    else:
        print("\n❌ Demo failed. Check installation and try again.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🚀 Ready to use with your real trades!")
    except KeyboardInterrupt:
        print("\n\n❌ Demo cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        print("\nTry running: pip install pandas numpy matplotlib seaborn scipy")
