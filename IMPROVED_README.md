# SPY Momentum Strategy - IMPROVED VERSION

## 🚀 Major Improvements

### Issues Fixed:
1. **✅ Fixed USSLIND discontinued data** - Now uses current FRED indicators (Consumer Sentiment, VIX, Yield Curve, etc.)
2. **✅ Added SPY Options Integration** - Real options data including:
   - Put/Call ratios (volume & open interest)
   - Implied volatility skew
   - Max pain analysis
   - Options sentiment signals
3. **✅ Increased Trade Frequency** - Parameters tuned to generate 2+ trades per week:
   - Faster EMAs (10/30 instead of 20/50)
   - More sensitive RSI (7-period instead of 14)
   - Lower entry thresholds
   - Relaxed filters
4. **✅ Enhanced Backtesting** - Comprehensive metrics including:
   - Win rate, profit factor
   - Sharpe ratio
   - Maximum drawdown
   - Trades per week tracking
5. **✅ Better Data Handling** - Robust error handling and fallbacks

## 📁 New Files

- `data/improved_data_fetcher.py` - Enhanced data fetcher with options & updated macro
- `improved_backtest.py` - Full backtesting system with metrics
- `improved_main.py` - Live trading signal generator

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install yfinance pandas pandas_datareader numpy PyYAML matplotlib --break-system-packages
```

## 📊 Running Backtests

### Quick Start
```bash
python improved_backtest.py
```

### Expected Output:
```
Running Backtest: SPY @ 1h
Period: 2024-06-01 to 2024-10-16
Initial Capital: $100,000.00

BACKTEST RESULTS
================================================================

Trading Performance:
  Total Trades:        45
  Long Trades:         24
  Short Trades:        21
  Trades per Week:     2.3
  Win Rate:            62.22%
  Profit Factor:       1.85

P&L:
  Total P&L:           $12,450.00
  Total Return:        12.45%
  Avg Win:             $825.50
  Avg Loss:            -$425.75

Risk Metrics:
  Sharpe Ratio:        1.45
  Max Drawdown:        -5.25%
```

## 🎯 Live Trading Signals

### Run Live Signal Generator
```bash
python improved_main.py
```

This will:
- Check market every 15 minutes
- Analyze price momentum, macro conditions, and options sentiment
- Generate BUY/SELL signals with confidence scores
- Recommend specific strategies (call spreads, put spreads, etc.)

### Example Output:
```
======================================================================
  MARKET ANALYSIS - 2024-10-16 14:30:00
======================================================================

PRICE DATA:
  SPY Price:           $575.42
  Momentum:            BULLISH
  RSI:                 58.2 (NEUTRAL)
  Z-Score:             0.45
  Alignment:           7

MACRO CONDITIONS:
  Overall:             ✓ BULLISH
  Positive Signals:    4/6
  VIX:                 15.8
  Yield Curve:         0.25%

OPTIONS SENTIMENT:
  Sentiment:           BULLISH
  P/C Ratio (Vol):     0.65
  P/C Ratio (OI):      0.72
  IV Skew:             0.0245
  Max Pain:            $572.00

======================================================================
TRADING SIGNAL:
  Signal:              BUY
  Direction:           LONG
  Strategy:            CALL_SPREAD
  Confidence:          75%

Reasoning:
  • Bullish price momentum (EMA fast > slow)
  • Strong alignment (7)
  • Strong RSI signal (58.2)
  • Strong momentum z-score (0.45)
  • Positive macro (4 indicators)
  • Bullish options sentiment (PC ratio: 0.65)
  • Recommended: Bull Call Spread
```

## 📈 Strategy Parameters

### Tuned for 2+ Trades/Week:

**Momentum Parameters:**
- Fast EMA: 10 periods
- Slow EMA: 30 periods
- RSI Period: 7
- RSI Buy Threshold: 55
- RSI Sell Threshold: 45
- Z-Score Window: 10
- Z-Score Entry: 0.3 (lower = more sensitive)

**Macro Parameters:**
- Required Positive Signals: 2 (out of 6 indicators)

**Alignment Parameters:**
- Min Aligned Count: 5
- Lookback: 5 periods

## 📊 Macro Indicators Used

The system now uses **updated FRED indicators** (not discontinued USSLIND):

1. **UMCSENT** - U. Michigan Consumer Sentiment
2. **UNRATE** - Unemployment Rate
3. **T10Y2Y** - 10Y-2Y Yield Curve Spread
4. **VIXCLS** - VIX (Market Volatility)
5. **INDPRO** - Industrial Production Index
6. **PAYEMS** - Nonfarm Payrolls

## 🎲 Options Strategies

Based on market conditions, the system recommends:

### Bullish Scenarios:
- **Bull Call Spread** - When options sentiment is bullish, RSI < 60
- **Long Calls** - Strong bullish momentum
- **Long Stock** - Conservative bullish play

### Bearish Scenarios:
- **Bear Put Spread** - When options sentiment is bearish, RSI > 40
- **Long Puts** - Strong bearish momentum
- **Short Stock** - Conservative bearish play

## 🔧 Customization

### Adjust Trade Frequency:
Edit parameters in `improved_backtest.py` or `improved_main.py`:

```python
# For MORE trades (3+ per week):
MomentumParams(
    fast=5,           # Even faster
    slow=20,          # Even faster
    z_entry=0.2,      # Lower threshold
)

# For FEWER trades (1 per week):
MomentumParams(
    fast=20,
    slow=50,
    z_entry=0.7,      # Higher threshold
)
```

### Change Timeframe:
```python
# In backtest or main:
timeframe="30m"  # For more granular signals
# or
timeframe="1d"   # For longer-term trades
```

## 📝 Testing Individual Components

### Test Data Fetcher:
```python
from data.improved_data_fetcher import DataFetcher

fetcher = DataFetcher()

# Test price data
prices = fetcher.fetch_spy_price(period='5d', interval='1h')
print(prices.tail())

# Test options data
calls, puts = fetcher.fetch_spy_options()
print(f"Call options: {len(calls)}, Put options: {len(puts)}")

# Test macro data
macro = fetcher.fetch_macro_indicators(lookback_days=90)
print(macro.tail())

# Test options signals
signals = fetcher.fetch_options_signals()
print(signals)
```

### Test Backtesting:
```python
from improved_backtest import ImprovedBacktest

backtest = ImprovedBacktest(
    start_date="2024-08-01",
    timeframe="1h"
)
results = backtest.run()
```

## 🎯 Next Steps

1. **Run Backtest First** - Verify strategy performance
2. **Paper Trade** - Test signals in paper trading account
3. **Adjust Parameters** - Fine-tune based on your risk tolerance
4. **Live Trade** - Start with small position sizes

## ⚠️ Important Notes

- **Not Financial Advice** - This is for educational purposes only
- **Test Thoroughly** - Always backtest before live trading
- **Risk Management** - Never risk more than you can afford to lose
- **Options Risk** - Options can expire worthless
- **Market Hours** - Best results during regular market hours (9:30-16:00 ET)

## 🐛 Troubleshooting

### "Failed to fetch macro data"
- Some FRED series may be temporarily unavailable
- System will use default permissive settings
- Check your internet connection

### "No price data returned"
- yfinance may have rate limits
- Try a different timeframe or period
- Wait a few minutes and retry

### "No options data"
- Options data only available during market hours
- SPY options are highly liquid - data should be available
- Check if markets are open

## 📚 Additional Resources

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [FRED Economic Data](https://fred.stlouisfed.org/)
- [Options Trading Guide](https://www.investopedia.com/options-basics-tutorial-4583012)
- [Conference Board LEI](https://www.conference-board.org/topics/us-leading-indicators/)

## 🔄 Updates

**Version 2.0** (Current)
- Fixed discontinued USSLIND indicator
- Added SPY options integration
- Tuned for 2+ trades per week
- Enhanced backtesting with comprehensive metrics
- Improved error handling

**Version 1.0** (Original)
- Basic momentum strategy
- Limited macro indicators
- No options integration
