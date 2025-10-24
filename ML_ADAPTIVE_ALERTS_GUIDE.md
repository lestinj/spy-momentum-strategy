## 🎯 This is Your Production System!

This alert system **exactly matches** your backtested strategy that achieved:
- **155% CAGR** (2023-2025)
- **102% CAGR** (2020-2025)  
- **73% CAGR** (2015-2025)

## ✅ Key Features

### 1. **Dynamic Capital Management**
Automatically adjusts position sizes as you add/withdraw capital.

### 2. **Market Regime Detection**
Identifies 5 market conditions and adjusts accordingly:
- **LOW_VOL**: 2.4x leverage, aggressive positioning
- **HIGH_VOL**: 1.2x leverage, conservative  
- **TRENDING_UP**: 2.0x leverage, max 4 positions
- **TRENDING_DOWN**: 1.5x leverage, quick exits
- **NORMAL**: Standard parameters

### 3. **ML-Based Position Sizing**
- High confidence (>60%): 50% position
- Medium (30-60%): Scaled 10-50%
- Low (<30%): 10% position

## 📊 Quick Start

### Initial Setup

```bash
# Install requirements
pip install yfinance pandas numpy scikit-learn

# Set your initial capital
python ml_adaptive_v49_alerts.py --capital 100000

# Or add to existing capital
python ml_adaptive_v49_alerts.py --add-capital 20000
```

### Running Alerts

#### Single Check
```bash
python ml_adaptive_v49_alerts.py --once
```

#### Continuous Monitoring (recommended)
```bash
# Check every 5 minutes
python ml_adaptive_v49_alerts.py --interval 300

# With email alerts


```

## 📈 Understanding the Output

### Market Regime Display
```
📊 Market Regime: LOW_VOL
   Leverage: 2.4x
   Max Positions: 3
   Capital: $100,000
```

### Buy Signal Example
```
#1 🟢 NVDA - TREND_FOLLOW
──────────────────────────
ML Confidence: ████████░░ 78% (HIGH)
Current Price: $650.00
Market Regime: LOW_VOL
Leverage: 2.4x

📋 TRADING INSTRUCTIONS:
Position Size: 45% of capital
1. BUY 165 shares @ $650.00
2. Capital Required: $45,000
3. STOP LOSS: $585.00 (-10%)
4. TAKE PROFIT: $845.00 (+30%)
```

## 🔧 Capital Management

### Check Current Capital
```bash
python ml_adaptive_v49_alerts.py --capital
```

### Add Money (e.g., adding $10,000)
```bash
python ml_adaptive_v49_alerts.py --add-capital 10000
```

### Set New Total (e.g., reset to $50,000)
```bash
python ml_adaptive_v49_alerts.py --capital 50000
```

## 📁 Files Created

The system creates these files:
- `capital_config.json` - Tracks your capital
- `adaptive_positions.json` - Current positions
- `adaptive_ml_models/` - Trained ML models

## 🎯 Trading Rules

### Entry Rules (Automated Detection)
1. **TREND_FOLLOW**: RSI > 55, price above MAs
2. **PULLBACK**: RSI 45-55, uptrend intact

### Exit Rules (Automated Detection)
1. **Stop Loss**: Dynamic 5-10% based on regime
2. **Take Profit**: Dynamic 15-35% based on regime
3. **Time Exit**: 14 days max
4. **Regime Exit**: Quick profit in HIGH_VOL

### Position Sizing by Regime

| Regime | Leverage | Max Positions | Stop Loss | Take Profit |
|--------|----------|---------------|-----------|-------------|
| LOW_VOL | 2.4x | 3 | 6% | 30% |
| HIGH_VOL | 1.2x | 2 | 10% | 20% |
| TRENDING_UP | 2.0x | 4 | 10% | 35% |
| TRENDING_DOWN | 1.5x | 2 | 6% | 15% |
| NORMAL | 2.0x | 3 | 8% | 25% |

## 📊 Expected Performance

Based on backtesting, you should see:

### By Market Regime
- **LOW_VOL** (most common): Highest returns
- **HIGH_VOL** (rare): Highest per-trade profit
- **TRENDING_UP**: Good win rates
- **TRENDING_DOWN**: Conservative profits

### Overall Expectations
- **Win Rate**: 52-56%
- **Avg Winners**: 20-30% gains
- **Avg Losers**: 5-10% losses
- **Max Drawdown**: ~40% (manage accordingly)

## ⚠️ Risk Management

### Maximum Risk
With full deployment (3 positions at 2.4x leverage):
- **Total Exposure**: Up to 216% of capital
- **Max Loss per Position**: 10% of position = 21.6% of capital
- **Total Portfolio Risk**: ~65% in worst case

### Recommendations
1. Start with 50% of intended capital
2. Add more as you gain confidence
3. Reduce size in HIGH_VOL regimes
4. Take profits regularly

## 🔍 Monitoring

### Daily Checks
1. Run `--once` each morning
2. Check regime changes
3. Verify position sizes

### Position Tracking
The system saves all positions in `adaptive_positions.json`:
```json
{
  "positions": {
    "NVDA": {
      "entry_price": 650.00,
      "shares": 165,
      "capital_used": 45000,
      "regime": "LOW_VOL",
      "stop_loss": 0.06,
      "take_profit": 0.30
    }
  }
}
```

## 🚀 Live Trading Checklist

### Before Going Live
- [ ] Test with paper trading first
- [ ] Set initial capital correctly
- [ ] Verify email alerts working
- [ ] Understand regime parameters
- [ ] Have exit plan for drawdowns

### Daily Operations
- [ ] Check market regime
- [ ] Review any signals
- [ ] Execute trades as indicated
- [ ] Update positions file
- [ ] Monitor P&L

### Weekly Maintenance
- [ ] Review performance vs backtest
- [ ] Retrain ML models if needed
- [ ] Adjust capital if needed
- [ ] Check regime distribution

## 📞 Support

### Common Issues

**No signals appearing**
- Market regime might be HIGH_VOL (fewer signals)
- All position slots might be full
- Check that markets are open

**ML confidence always 50%**
- Models might need retraining
- Delete `adaptive_ml_models/` folder to retrain

**Capital mismatch**
- Run reconciliation: Check positions vs capital
- Update capital config if needed

## 💡 Pro Tips

1. **Best Times**: Most signals appear at market open
2. **Regime Changes**: Pay attention when regime shifts
3. **Capital Scaling**: Add capital during drawdowns
4. **Profit Taking**: Consider taking some profits at 20%+

## ⚡ Quick Commands Reference

```bash
# Check signals once
python ml_adaptive_v49_alerts.py --once

# Run continuously
python ml_adaptive_v49_alerts.py

# With email
python ml_adaptive_v49_alerts.py --email you@gmail.com --password xxx

# Update capital
python ml_adaptive_v49_alerts.py --add-capital 5000

# Check every 10 minutes
python ml_adaptive_v49_alerts.py --interval 600
```

---

**Remember**: This system achieved 155% CAGR in backtesting. Actual results may vary. Trade responsibly!
