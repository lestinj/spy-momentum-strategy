# V4.9 Backtest - Fixed Version

## What Was Fixed

### 1. **Date Format Issue**
- **Problem:** Script used `'24-01-01'` format (ambiguous)
- **Fix:** Now uses `'2024-01-01'` format (clear)
- **Bonus:** Auto-converts old format if you use it

### 2. **Data Loading Threshold**
- **Problem:** Required >100 days of data (too much for 3-6 month tests)
- **Fix:** Now requires >50 days (allows shorter backtests)

### 3. **No Error Handling**
- **Problem:** If no symbols loaded, script would crash later
- **Fix:** Now checks if data loaded and shows helpful error message

### 4. **Missing Data Cleanup**
- **Problem:** NaN values could cause issues
- **Fix:** Explicitly drops NaN rows after indicator calculation

### 5. **File Naming**
- **Problem:** Mixed v48/v49 naming
- **Fix:** Consistent V4.9 naming throughout

## How To Use

### Quick 3-Month Test
```python
python v49_backtest.py
# Default: Last 6 months (2024-04-01 to now)
```

### Custom Date Range
Edit the main section at bottom of file:

```python
# Last 3 months
strategy.load_data(start_date='2024-07-01')

# Last 6 months
strategy.load_data(start_date='2024-04-01')

# Year to date
strategy.load_data(start_date='2024-01-01')

# Full backtest (5 years)
strategy.load_data(start_date='2020-01-01')

# Specific range
strategy.load_data(start_date='2024-01-01', end_date='2024-06-30')
```

## Output Files

All files now use clean v49 naming:
- `v49_trades.csv` - All trades
- `v49_equity.csv` - Daily equity curve
- `v49_monthly.csv` - Monthly performance
- `v49_weekly.csv` - Weekly performance

## What It Tests

**Strategies:**
- TREND_FOLLOW (RSI > 55, price > both MAs)
- PULLBACK (RSI 45-55, price > slow MA)

**Parameters:**
- 3 positions @ 45% each
- 2.5x leverage
- 8% stop loss
- 25% take profit
- 14-day max hold

**Symbols:**
NVDA, TSLA, PLTR, AMD, COIN, META, NET

## Troubleshooting

### "Loaded 0 symbols"
- Check internet connection
- Verify date range has enough data (need 50+ days)
- Try updating yfinance: `pip install --upgrade yfinance`

### "IndexError: list index out of range"
- Usually means no data loaded
- Script now prevents this with early error check

### Partial symbol loading
- Normal - some symbols may have insufficient data for short ranges
- Script will run as long as at least 1 symbol loads

## Example Output

```
================================================================================
V4.9 BACKTEST
TREND_FOLLOW + PULLBACK strategies
================================================================================

📊 Loading V4.9 data with 2 WINNING strategies...
Period: 2024-04-01 to 2025-10-18

✓ NVDA: 142 days
✓ TSLA: 142 days
✓ PLTR: 142 days
✓ AMD: 142 days
✓ COIN: 142 days
✓ META: 142 days
✓ NET: 142 days

✓ Loaded 7 symbols

================================================================================
V4.9 - WINNERS ONLY
Only TREND_FOLLOW and PULLBACK strategies
================================================================================

📅 Period: 2024-04-01 to 2025-10-17
💰 Initial: $10,000

[... trades execute ...]

================================================================================
BACKTEST RESULTS - V4.9 WINNERS ONLY
================================================================================
Initial Capital:    $         10,000
Final Equity:       $         15,420
Total Return:                  54.2%
CAGR:                          89.3%
Max Drawdown:                 -23.4%

Total Trades:                     23
Winners:                          12 (52.2%)
Losers:                           11
Avg Win:            $           842
Avg Loss:           $          -421
Avg Hold:                      11.3 days
Win/Loss Ratio:                2.00x
================================================================================

📁 Files saved:
   • v49_trades.csv - All 23 trades
   • v49_equity.csv
   • v49_monthly.csv
   • v49_weekly.csv

✅ Complete!
```

## Notes

- Shorter backtests (3-6 months) will have fewer trades
- Performance will vary by date range selected
- Full 5-year backtest gives most reliable results
- Use short tests to validate strategy is working, not for final performance numbers
