# V51 Hybrid Strategy - Quick Reference Guide

## 🎯 Strategy Overview

**V51 Hybrid** combines the best elements of two proven approaches:
- **V48's Setup**: 195% CAGR with 3 positions, static allocation
- **V50's Filters**: Quality trade selection for improved win rate

## 📊 Key Parameters

### Position Management
- **Max Positions**: 3 simultaneous trades
- **Position Size**: 33.33% per position (static allocation)
- **Initial Capital**: $30,000

### Entry Rules
- **Primary Signal**: RSI ≤ 30 (oversold)
- **Quality Filters**:
  - Minimum volume: 1,000,000 shares/day
  - Price range: $5 - $500
  - Volume spike: 1.5x average (50% above normal)
  - RSI confirmation: 3 days of oversold readings
  - Volatility filter: ATR < 5%
  - Momentum check: Not falling more than 10% in 5 days

### Exit Rules
- **Stop Loss**: 8% below entry
- **Take Profit**: 15% above entry
- **RSI Exit**: RSI ≥ 70 (overbought)

## 🚀 How to Run

### Basic Usage
```python
from v51_hybrid_strategy import V51Hybrid

# Initialize strategy
strategy = V51Hybrid(initial_capital=30000)

# Load data (will download from Yahoo Finance or use synthetic data)
strategy.load_data(start_date='2024-01-01')

# Run backtest
results = strategy.run_backtest()
```

### Custom Parameters
```python
# Modify parameters before running
strategy = V51Hybrid(initial_capital=50000)
strategy.max_positions = 4  # Try 4 positions
strategy.rsi_entry = 25  # More aggressive entry
strategy.stop_loss_pct = 0.10  # Wider stops
```

## 📈 Expected Results

### Target Metrics
- **CAGR**: 195%+ (match or exceed V48)
- **Win Rate**: 50%+ (improvement from quality filters)
- **Max Drawdown**: < 30%
- **Profit Factor**: > 2.0

### What V48 Achieved (Baseline)
- **CAGR**: 195%
- **Setup**: 3 positions, static allocation
- **Weakness**: No quality filters (trades all signals)

### What V51 Adds (Improvements)
- ✅ Volume confirmation (avoid low-liquidity trades)
- ✅ Price range filters (avoid penny stocks and too-expensive stocks)
- ✅ RSI confirmation (avoid false signals)
- ✅ Volatility filter (avoid getting stopped out too easily)
- ✅ Momentum validation (ensure recovery is starting)

## 📁 Output Files

After running, you'll get:
1. **v51_hybrid_trades.csv** - Detailed trade log
2. **v51_hybrid_equity.csv** - Daily equity curve

### Trades CSV Columns
- symbol, entry_date, entry_price, exit_date, exit_price
- shares, pnl, pnl_pct, exit_reason, hold_days

### Equity CSV Columns
- date, equity, positions, cash

## 🔍 Interpreting Results

### Success Indicators
✅ **CAGR ≥ 195%**: Strategy matches V48 baseline
✅ **Win Rate ≥ 50%**: Quality filters working
✅ **Rejection Rate 30-50%**: Filters are selective but not too strict
✅ **Profit Factor > 2.0**: Winners significantly outweigh losers

### Warning Signs
⚠️ **CAGR < 180%**: Filters may be too strict
⚠️ **Win Rate < 45%**: Quality filters not helping enough
⚠️ **Rejection Rate > 70%**: Missing too many opportunities
⚠️ **Max DD > 35%**: Risk too high

## 🎛️ Tuning Guide

### If Win Rate is Low (< 45%)
- Tighten quality filters:
  - Increase `volume_spike_threshold` to 2.0
  - Increase `rsi_confirm_period` to 5
  - Lower `max_volatility` threshold

### If CAGR is Low (< 180%)
- Loosen quality filters:
  - Decrease `volume_spike_threshold` to 1.3
  - Decrease `rsi_confirm_period` to 2
  - Widen stop loss to 10%

### If Max Drawdown is High (> 35%)
- Reduce risk:
  - Decrease `max_positions` to 2
  - Tighten stop loss to 6%
  - Increase quality filter strictness

## 📊 Quality Filter Impact

The strategy reports how many signals were:
- **Accepted**: Passed all quality filters
- **Rejected**: Failed one or more filters

**Ideal Rejection Rate**: 30-50%
- Too low (< 20%): Filters not selective enough
- Too high (> 70%): Missing good opportunities

## 🔄 Comparison Workflow

1. **Run V51 Hybrid** (this strategy)
2. **Compare against V48 baseline**:
   - Is CAGR similar or better?
   - Is win rate improved?
   - Is drawdown acceptable?
3. **Decision**:
   - If V51 > V48: Use V51 for paper trading
   - If V51 ≈ V48: Both are viable, choose based on preference
   - If V51 < V48: Tune parameters or revert to V48

## 🎯 Next Steps After Running

1. **Review Console Output**
   - Check CAGR vs 195% target
   - Check win rate vs 50% target
   - Review quality filter acceptance rate

2. **Analyze Trade Details**
   - Open `v51_hybrid_trades.csv`
   - Look for patterns in winners/losers
   - Check exit reasons distribution

3. **Examine Equity Curve**
   - Open `v51_hybrid_equity.csv`
   - Plot equity over time
   - Identify drawdown periods

4. **Make Decision**
   - If successful: Move to paper trading
   - If needs work: Tune parameters and re-run
   - If unsuccessful: Consider reverting to V48

## ⚙️ Advanced Customization

### Add Your Own Symbols
```python
strategy = V51Hybrid(initial_capital=30000)
strategy.symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA']  # Custom list
```

### Adjust Risk Parameters
```python
strategy.stop_loss_pct = 0.12  # 12% stop
strategy.take_profit_pct = 0.20  # 20% target
```

### Modify Quality Filters
```python
strategy.min_volume = 2000000  # Higher liquidity requirement
strategy.volume_spike_threshold = 2.0  # Stronger volume confirmation
strategy.rsi_confirm_period = 5  # More confirmation
```

## 📞 Quick Troubleshooting

### "No trades generated"
- Quality filters may be too strict
- Try loosening filters or checking data quality

### "Win rate still low"
- Filters may not be strict enough
- Consider tightening quality criteria

### "CAGR much lower than V48"
- Filters rejecting too many good trades
- Try reducing filter strictness

### "Max drawdown too high"
- Reduce position size or max positions
- Tighten stop loss

## 🏁 Success Checklist

Before considering V51 for paper trading:
- [ ] CAGR ≥ 195% (or within 10% of V48)
- [ ] Win rate ≥ 50%
- [ ] Max drawdown < 30%
- [ ] Profit factor > 2.0
- [ ] Quality filters rejecting 30-50% of signals
- [ ] Results make sense and are explainable
- [ ] Backtested on sufficient data (1+ years)

## 💡 Key Insights

**What makes V51 different?**
- Same proven position structure as V48
- Adds intelligent filtering to avoid bad trades
- Aims to maintain returns while improving win rate

**Philosophy:**
- Quality over quantity
- Protect V48's performance
- Improve risk-adjusted returns
- Make trading more consistent

**Remember:**
If V51 doesn't beat or match V48, that's okay! V48 already performs well. The goal is improvement, not perfection.
