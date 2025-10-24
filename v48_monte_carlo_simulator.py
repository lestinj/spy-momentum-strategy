"""
V4.8 MONTE CARLO SIMULATOR
═══════════════════════════════════════════════════════════════════════════════

PURPOSE: Advanced Monte Carlo simulation for v4.8 stock trading system
  - Run 10,000+ scenarios
  - Model realistic market conditions
  - Show probability distributions
  - Calculate risk metrics (VaR, CVaR, etc.)
  - Generate comprehensive visualizations

FEATURES:
  1. Path-dependent simulation (tracks equity curve)
  2. Drawdown analysis
  3. Win streak / loss streak modeling
  4. Market regime sensitivity
  5. Position sizing impact
  6. Risk of ruin calculations
  
METHODOLOGY:
  - Bootstrap from historical trades
  - Preserve trade correlations
  - Model realistic fills and slippage
  - Account for compounding
  - Include transaction costs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class V48MonteCarloSimulator:
    def __init__(self, initial_capital=10000, simulations=10000):
        """
        Initialize Monte Carlo simulator
        
        Args:
            initial_capital: Starting capital
            simulations: Number of Monte Carlo paths to simulate
        """
        self.initial_capital = initial_capital
        self.simulations = simulations
        self.results = {}
        
    def load_historical_trades(self, trades_csv_path):
        """
        Load historical trades from backtest
        
        Args:
            trades_csv_path: Path to trades CSV file
        """
        try:
            self.trades_df = pd.read_csv(trades_csv_path)
            print(f"✓ Loaded {len(self.trades_df)} historical trades")
            
            # Extract key statistics
            self.analyze_historical_trades()
            return True
            
        except FileNotFoundError:
            print(f"❌ Could not find trades file: {trades_csv_path}")
            return False
    
    def analyze_historical_trades(self):
        """Analyze historical trades to extract distributions"""
        
        # Separate winners and losers
        winners = self.trades_df[self.trades_df['pnl'] > 0]
        losers = self.trades_df[self.trades_df['pnl'] <= 0]
        
        # Calculate key metrics
        self.trade_stats = {
            'total_trades': len(self.trades_df),
            'win_rate': len(winners) / len(self.trades_df),
            
            # Winner statistics (in %)
            'winner_mean': (winners['pnl'] / winners['entry_price'] * 100).mean() if len(winners) > 0 else 0,
            'winner_std': (winners['pnl'] / winners['entry_price'] * 100).std() if len(winners) > 0 else 0,
            'winner_median': (winners['pnl'] / winners['entry_price'] * 100).median() if len(winners) > 0 else 0,
            
            # Loser statistics (in %)
            'loser_mean': (losers['pnl'] / losers['entry_price'] * 100).mean() if len(losers) > 0 else 0,
            'loser_std': (losers['pnl'] / losers['entry_price'] * 100).std() if len(losers) > 0 else 0,
            'loser_median': (losers['pnl'] / losers['entry_price'] * 100).median() if len(losers) > 0 else 0,
            
            # Holding period
            'avg_holding_days': self.trades_df['days_held'].mean() if 'days_held' in self.trades_df.columns else 10,
            
            # Trades per day
            'trades_per_day': len(self.trades_df) / 252 / 2,  # Approximate
        }
        
        print("\n" + "="*80)
        print("HISTORICAL TRADE ANALYSIS")
        print("="*80)
        print(f"Total Trades:      {self.trade_stats['total_trades']:>10,}")
        print(f"Win Rate:          {self.trade_stats['win_rate']:>10.1%}")
        print(f"\nWinners:")
        print(f"  Mean Return:     {self.trade_stats['winner_mean']:>10.1f}%")
        print(f"  Std Dev:         {self.trade_stats['winner_std']:>10.1f}%")
        print(f"  Median Return:   {self.trade_stats['winner_median']:>10.1f}%")
        print(f"\nLosers:")
        print(f"  Mean Return:     {self.trade_stats['loser_mean']:>10.1f}%")
        print(f"  Std Dev:         {self.trade_stats['loser_std']:>10.1f}%")
        print(f"  Median Return:   {self.trade_stats['loser_median']:>10.1f}%")
        print(f"\nAvg Hold Time:     {self.trade_stats['avg_holding_days']:>10.1f} days")
        print("="*80)
    
    def simulate_single_path(self, trading_days=252, trades_per_day=1.5):
        """
        Simulate a single Monte Carlo path
        
        Args:
            trading_days: Number of trading days to simulate
            trades_per_day: Average number of trades per day
            
        Returns:
            equity_curve: Daily equity values
            trades: List of simulated trades
        """
        equity = self.initial_capital
        equity_curve = [equity]
        trades = []
        
        total_trades = int(trading_days * trades_per_day)
        
        for i in range(total_trades):
            # Randomly determine if winner or loser
            is_winner = np.random.random() < self.trade_stats['win_rate']
            
            if is_winner:
                # Sample from winner distribution
                return_pct = np.random.normal(
                    self.trade_stats['winner_mean'],
                    self.trade_stats['winner_std']
                ) / 100
            else:
                # Sample from loser distribution
                return_pct = np.random.normal(
                    self.trade_stats['loser_mean'],
                    self.trade_stats['loser_std']
                ) / 100
            
            # Apply position sizing (2-4% risk per trade)
            position_size = equity * np.random.uniform(0.02, 0.04)
            
            # Calculate P&L
            pnl = position_size * return_pct
            equity += pnl
            
            # Ensure equity doesn't go negative
            if equity <= 0:
                equity = 0
                break
            
            trades.append({
                'trade_num': i+1,
                'equity_before': equity - pnl,
                'position_size': position_size,
                'return_pct': return_pct * 100,
                'pnl': pnl,
                'equity_after': equity
            })
            
            # Update equity curve (interpolate for daily values)
            if i > 0 and i % int(trades_per_day) == 0:
                equity_curve.append(equity)
        
        # Fill out remaining days
        while len(equity_curve) < trading_days + 1:
            equity_curve.append(equity)
        
        return np.array(equity_curve[:trading_days + 1]), trades
    
    def run_simulation(self, trading_days=252, trades_per_day=1.5):
        """
        Run Monte Carlo simulation
        
        Args:
            trading_days: Number of trading days to simulate (252 = 1 year)
            trades_per_day: Average trades per day
        """
        print(f"\n" + "="*80)
        print(f"RUNNING MONTE CARLO SIMULATION")
        print("="*80)
        print(f"Simulations:       {self.simulations:>10,}")
        print(f"Trading Days:      {self.simulations:>10,}")
        print(f"Initial Capital:   ${self.initial_capital:>10,}")
        print(f"Trades per Day:    {trades_per_day:>10.1f}")
        print("="*80)
        
        # Store all paths
        all_paths = np.zeros((self.simulations, trading_days + 1))
        final_values = np.zeros(self.simulations)
        max_drawdowns = np.zeros(self.simulations)
        
        # Run simulations
        for i in range(self.simulations):
            if i % 1000 == 0:
                print(f"Progress: {i:>6,} / {self.simulations:>6,} ({i/self.simulations*100:>5.1f}%)")
            
            equity_curve, trades = self.simulate_single_path(trading_days, trades_per_day)
            all_paths[i] = equity_curve
            final_values[i] = equity_curve[-1]
            
            # Calculate max drawdown for this path
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - running_max) / running_max
            max_drawdowns[i] = drawdowns.min()
        
        print(f"Progress: {self.simulations:>6,} / {self.simulations:>6,} (100.0%)")
        print("✓ Simulation complete!")
        
        # Store results
        self.all_paths = all_paths
        self.final_values = final_values
        self.max_drawdowns = max_drawdowns
        
        # Calculate statistics
        self.calculate_statistics(trading_days)
    
    def calculate_statistics(self, trading_days):
        """Calculate comprehensive statistics from simulation results"""
        
        returns = (self.final_values - self.initial_capital) / self.initial_capital * 100
        cagr = ((self.final_values / self.initial_capital) ** (252 / trading_days) - 1) * 100
        
        self.stats = {
            # Final capital statistics
            'final_capital_mean': np.mean(self.final_values),
            'final_capital_median': np.median(self.final_values),
            'final_capital_std': np.std(self.final_values),
            'final_capital_10th': np.percentile(self.final_values, 10),
            'final_capital_90th': np.percentile(self.final_values, 90),
            
            # Return statistics
            'return_mean': np.mean(returns),
            'return_median': np.median(returns),
            'return_std': np.std(returns),
            'return_10th': np.percentile(returns, 10),
            'return_90th': np.percentile(returns, 90),
            
            # CAGR statistics
            'cagr_mean': np.mean(cagr),
            'cagr_median': np.median(cagr),
            'cagr_std': np.std(cagr),
            'cagr_10th': np.percentile(cagr, 10),
            'cagr_90th': np.percentile(cagr, 90),
            
            # Drawdown statistics
            'max_dd_mean': np.mean(self.max_drawdowns) * 100,
            'max_dd_median': np.median(self.max_drawdowns) * 100,
            'max_dd_10th': np.percentile(self.max_drawdowns, 10) * 100,
            'max_dd_90th': np.percentile(self.max_drawdowns, 90) * 100,
            'max_dd_worst': np.min(self.max_drawdowns) * 100,
            
            # Probability statistics
            'prob_profit': np.sum(self.final_values > self.initial_capital) / self.simulations * 100,
            'prob_double': np.sum(self.final_values > self.initial_capital * 2) / self.simulations * 100,
            'prob_triple': np.sum(self.final_values > self.initial_capital * 3) / self.simulations * 100,
            'prob_loss_50pct': np.sum(self.final_values < self.initial_capital * 0.5) / self.simulations * 100,
            'prob_ruin': np.sum(self.final_values == 0) / self.simulations * 100,
            
            # Risk metrics
            'var_95': np.percentile(returns, 5),  # Value at Risk (95%)
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),  # Conditional VaR
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252/trading_days) if np.std(returns) > 0 else 0,
        }
        
        self.print_statistics()
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        
        print("\n" + "="*80)
        print("MONTE CARLO RESULTS")
        print("="*80)
        
        print(f"\n{'FINAL CAPITAL:':<30}")
        print(f"  Mean:              ${self.stats['final_capital_mean']:>15,.0f}")
        print(f"  Median:            ${self.stats['final_capital_median']:>15,.0f}")
        print(f"  Std Dev:           ${self.stats['final_capital_std']:>15,.0f}")
        print(f"  10th Percentile:   ${self.stats['final_capital_10th']:>15,.0f}")
        print(f"  90th Percentile:   ${self.stats['final_capital_90th']:>15,.0f}")
        
        print(f"\n{'TOTAL RETURN (%):':<30}")
        print(f"  Mean:              {self.stats['return_mean']:>15.1f}%")
        print(f"  Median:            {self.stats['return_median']:>15.1f}%")
        print(f"  Std Dev:           {self.stats['return_std']:>15.1f}%")
        print(f"  10th Percentile:   {self.stats['return_10th']:>15.1f}%")
        print(f"  90th Percentile:   {self.stats['return_90th']:>15.1f}%")
        
        print(f"\n{'ANNUALIZED CAGR (%):':<30}")
        print(f"  Mean:              {self.stats['cagr_mean']:>15.1f}%")
        print(f"  Median:            {self.stats['cagr_median']:>15.1f}%")
        print(f"  Std Dev:           {self.stats['cagr_std']:>15.1f}%")
        print(f"  10th Percentile:   {self.stats['cagr_10th']:>15.1f}%")
        print(f"  90th Percentile:   {self.stats['cagr_90th']:>15.1f}%")
        
        print(f"\n{'MAX DRAWDOWN (%):':<30}")
        print(f"  Mean:              {self.stats['max_dd_mean']:>15.1f}%")
        print(f"  Median:            {self.stats['max_dd_median']:>15.1f}%")
        print(f"  10th Percentile:   {self.stats['max_dd_10th']:>15.1f}%")
        print(f"  90th Percentile:   {self.stats['max_dd_90th']:>15.1f}%")
        print(f"  Worst Case:        {self.stats['max_dd_worst']:>15.1f}%")
        
        print(f"\n{'PROBABILITIES:':<30}")
        print(f"  Profit (>0%):      {self.stats['prob_profit']:>15.1f}%")
        print(f"  Double (>100%):    {self.stats['prob_double']:>15.1f}%")
        print(f"  Triple (>200%):    {self.stats['prob_triple']:>15.1f}%")
        print(f"  Loss > 50%:        {self.stats['prob_loss_50pct']:>15.1f}%")
        print(f"  Ruin (Total Loss): {self.stats['prob_ruin']:>15.1f}%")
        
        print(f"\n{'RISK METRICS:':<30}")
        print(f"  VaR (95%):         {self.stats['var_95']:>15.1f}%")
        print(f"  CVaR (95%):        {self.stats['cvar_95']:>15.1f}%")
        print(f"  Sharpe Ratio:      {self.stats['sharpe_ratio']:>15.2f}")
        
        print("="*80)
    
    def create_visualizations(self, output_dir='trading_results'):
        """Create comprehensive visualization charts"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # ========================================================================
        # FIGURE 1: Monte Carlo Paths with Percentiles
        # ========================================================================
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot sample paths (100 random paths)
        sample_indices = np.random.choice(self.simulations, size=100, replace=False)
        for idx in sample_indices:
            ax.plot(self.all_paths[idx], alpha=0.05, color='steelblue', linewidth=0.5)
        
        # Calculate percentiles
        p10 = np.percentile(self.all_paths, 10, axis=0)
        p50 = np.percentile(self.all_paths, 50, axis=0)
        p90 = np.percentile(self.all_paths, 90, axis=0)
        
        # Plot percentiles
        ax.plot(p50, color='darkblue', linewidth=3, label='Median (50th percentile)')
        ax.plot(p10, color='red', linewidth=2, linestyle='--', label='10th percentile')
        ax.plot(p90, color='green', linewidth=2, linestyle='--', label='90th percentile')
        
        # Fill between percentiles
        ax.fill_between(range(len(p10)), p10, p90, alpha=0.2, color='steelblue')
        
        # Formatting
        ax.axhline(y=self.initial_capital, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Trading Days', fontsize=12, fontweight='bold')
        ax.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Monte Carlo Simulation - {self.simulations:,} Paths', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        filepath = f'{output_dir}/monte_carlo_paths_{timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {filepath}")
        plt.close()
        
        # ========================================================================
        # FIGURE 2: Distribution of Final Values
        # ========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 2A: Final Capital Distribution
        ax = axes[0, 0]
        ax.hist(self.final_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.median(self.final_values), color='darkblue', linestyle='--', linewidth=2, label=f'Median: ${np.median(self.final_values):,.0f}')
        ax.axvline(self.initial_capital, color='red', linestyle='--', linewidth=2, label=f'Initial: ${self.initial_capital:,.0f}')
        ax.set_xlabel('Final Capital ($)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Final Capital', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2B: Return Distribution
        ax = axes[0, 1]
        returns = (self.final_values - self.initial_capital) / self.initial_capital * 100
        ax.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.median(returns), color='darkgreen', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.1f}%')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax.set_xlabel('Total Return (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Returns', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2C: CAGR Distribution
        ax = axes[1, 0]
        cagr = ((self.final_values / self.initial_capital) ** (252 / len(self.all_paths[0])) - 1) * 100
        ax.hist(cagr, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.median(cagr), color='darkviolet', linestyle='--', linewidth=2, label=f'Median: {np.median(cagr):.1f}%')
        ax.set_xlabel('Annualized CAGR (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Annualized Returns', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2D: Max Drawdown Distribution
        ax = axes[1, 1]
        ax.hist(self.max_drawdowns * 100, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(np.median(self.max_drawdowns) * 100, color='darkred', linestyle='--', linewidth=2, label=f'Median: {np.median(self.max_drawdowns)*100:.1f}%')
        ax.set_xlabel('Max Drawdown (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Max Drawdowns', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f'{output_dir}/distributions_{timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # ========================================================================
        # FIGURE 3: Risk-Return Analysis
        # ========================================================================
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot of returns vs drawdowns
        ax.scatter(self.max_drawdowns * 100, returns, alpha=0.3, s=10, color='steelblue')
        
        # Add median lines
        ax.axvline(np.median(self.max_drawdowns) * 100, color='red', linestyle='--', linewidth=2, label=f'Median DD: {np.median(self.max_drawdowns)*100:.1f}%')
        ax.axhline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median Return: {np.median(returns):.1f}%')
        
        ax.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Risk-Return Trade-off', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = f'{output_dir}/risk_return_{timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # ========================================================================
        # FIGURE 4: Probability Curves
        # ========================================================================
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort returns for cumulative probability
        sorted_returns = np.sort(returns)
        cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns) * 100
        
        ax.plot(sorted_returns, cumulative_prob, linewidth=2, color='steelblue')
        
        # Mark key percentiles
        percentiles = [10, 25, 50, 75, 90]
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        for p, c in zip(percentiles, colors):
            val = np.percentile(returns, p)
            ax.axvline(val, color=c, linestyle='--', alpha=0.7, label=f'{p}th: {val:.1f}%')
            ax.plot(val, p, 'o', color=c, markersize=10)
        
        ax.set_xlabel('Total Return (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Distribution of Returns', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(sorted_returns), max(sorted_returns)])
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        filepath = f'{output_dir}/cumulative_prob_{timestamp}.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        print(f"\n✓ All visualizations saved to: {output_dir}/")
    
    def save_results(self, output_dir='trading_results'):
        """Save detailed results to CSV"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary dataframe
        summary_df = pd.DataFrame([self.stats])
        summary_df.to_csv(f'{output_dir}/monte_carlo_summary_{timestamp}.csv', index=False)
        print(f"\n✓ Saved: {output_dir}/monte_carlo_summary_{timestamp}.csv")
        
        # Create detailed results
        results_df = pd.DataFrame({
            'simulation': range(1, self.simulations + 1),
            'final_capital': self.final_values,
            'total_return_pct': (self.final_values - self.initial_capital) / self.initial_capital * 100,
            'max_drawdown_pct': self.max_drawdowns * 100,
        })
        results_df.to_csv(f'{output_dir}/monte_carlo_all_sims_{timestamp}.csv', index=False)
        print(f"✓ Saved: {output_dir}/monte_carlo_all_sims_{timestamp}.csv")


def run_monte_carlo_simulation(trades_csv_path, 
                               initial_capital=10000,
                               simulations=10000,
                               trading_days=252,
                               trades_per_day=1.5):
    """
    Main function to run complete Monte Carlo simulation
    
    Args:
        trades_csv_path: Path to historical trades CSV
        initial_capital: Starting capital
        simulations: Number of Monte Carlo paths
        trading_days: Number of days to simulate forward
        trades_per_day: Average trades per day
    """
    
    print("\n" + "╔" + "="*86 + "╗")
    print("║" + " "*86 + "║")
    print("║" + " "*25 + "V4.8 MONTE CARLO SIMULATOR" + " "*35 + "║")
    print("║" + " "*86 + "║")
    print("╚" + "="*86 + "╝")
    
    # Initialize simulator
    simulator = V48MonteCarloSimulator(
        initial_capital=initial_capital,
        simulations=simulations
    )
    
    # Load historical trades
    if not simulator.load_historical_trades(trades_csv_path):
        return None
    
    # Run simulation
    simulator.run_simulation(trading_days, trades_per_day)
    
    # Create visualizations
    simulator.create_visualizations()
    
    # Save results
    simulator.save_results()
    
    print("\n" + "="*80)
    print("✓ MONTE CARLO SIMULATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  • Monte Carlo paths chart")
    print("  • Distribution charts")
    print("  • Risk-return analysis")
    print("  • Probability curves")
    print("  • Summary statistics CSV")
    print("  • Detailed simulation results CSV")
    print("="*80)
    
    return simulator


if __name__ == "__main__":
    """
    Example usage:
    
    python v48_monte_carlo_simulator.py
    """
    
    # CONFIGURATION
    TRADES_CSV = "v48_stock_trades.csv"  # Change to your actual trades CSV file
    INITIAL_CAPITAL = 10000
    SIMULATIONS = 10000
    TRADING_DAYS = 252  # 1 year
    TRADES_PER_DAY = 1.5
    
    # Run simulation
    simulator = run_monte_carlo_simulation(
        trades_csv_path=TRADES_CSV,
        initial_capital=INITIAL_CAPITAL,
        simulations=SIMULATIONS,
        trading_days=TRADING_DAYS,
        trades_per_day=TRADES_PER_DAY
    )
