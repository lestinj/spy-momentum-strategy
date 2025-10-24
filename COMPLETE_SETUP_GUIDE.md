## 🎯 Your Trading Setup Tools

### 📊 1. Trading Dashboard
Complete view of positions, take profits, and rebalancing needs.

```bash
python trading_dashboard.py
```

Shows:
- Current P&L for each position
- Take profit orders to set
- Stop loss verification
- Rebalancing recommendations
- Risk metrics

### 📈 2. Take Profit Calculator
Generates exact take profit orders for your positions.

```bash
python setup_take_profits.py
```

**Your Current Take Profit Levels:**
```
AMD:  $301.39 (30% from $231.84)
TSLA: $571.73 (30% from $439.79)
NVDA: $238.76 (30% from $183.66)
```

**To Set in Your Broker:**
1. AMD: LIMIT SELL 14 shares @ $301.39 GTC
2. TSLA: LIMIT SELL 35 shares @ $571.73 GTC
3. NVDA: LIMIT SELL 65 shares @ $238.76 GTC

*GTC = Good Till Cancelled*

### 🛑 Stop Loss Verification
Verify these are set:
```
AMD:  STOP @ $218.13 (-6%)
TSLA: STOP @ $413.60 (-6%)
NVDA: STOP @ $172.64 (-6%)
```

## ⏰ 3. Auto-Start Setup

### For Windows:
```bash
setup_windows_autostart.bat
```

1. Edit the paths in `ml_v49_morning_check.bat`
2. Import `ML_V49_Task.xml` into Task Scheduler
3. Set to run at 9:00 AM every weekday

### For Mac/Linux:
```bash
./setup_linux_autostart.sh
```

1. Edit paths in `~/ml_v49_morning_check.sh`
2. Add to crontab: `crontab -e`
3. Add line: `0 9 * * 1-5 $HOME/ml_v49_morning_check.sh`

### Schedule Options:
- **9:00 AM**: Pre-market check
- **9:30 AM**: Market open check
- **12:00 PM**: Midday check
- **3:30 PM**: Near close check

## 🔄 4. Tomorrow's Rebalancing Plan

Based on your current positions:

### Current Allocation:
```
AMD:  $3,246  (20% of capital) → Under-allocated
TSLA: $15,393 (96% of capital) → Over-allocated
NVDA: $11,938 (75% of capital) → Slightly under
Total: $30,577 (191% of capital, 1.91x leverage)
```

### Target Allocation:
```
Each position: $12,800 (80% of capital)
Total: $38,400 (240% of capital, 2.4x leverage)
```

### Rebalancing Actions:
```
1. SELL 5-6 TSLA shares (~$2,600)
2. BUY 41 AMD shares (~$9,500)
3. Optional: BUY 5 NVDA shares (~$900)
```

## 📋 Daily Workflow

### Morning (9:00 AM):
1. **Automated check** runs via Task Scheduler/cron
2. Review any alerts
3. Check `trading_dashboard.py` for current P&L

### During Market Hours:
1. Execute any rebalancing trades
2. Set take profit orders if not already set
3. Verify stop losses are active

### End of Day:
1. Note positions that hit stops or take profits
2. Update `fix_positions_now.py` with changes
3. Run evening check for next day's signals

## 💾 File Organization

Create this folder structure:
```
trading/
├── ml_adaptive_v49_alerts_fixed.py   (main alerts)
├── trading_dashboard.py               (position monitor)
├── setup_take_profits.py              (TP calculator)
├── fix_positions_now.py               (position updater)
├── capital_config.json                (capital tracking)
├── adaptive_positions.json            (position tracking)
├── logs/                              (daily logs)
│   └── ml_v49_20251020.log
└── ml_v49_morning_check.bat/sh       (auto-start script)
```

## ⚠️ Important Reminders

### Risk Management:
- **Max Loss**: ~$1,800 if all stops hit (11% of capital)
- **Target Profit**: ~$9,200 if all TPs hit (57% of capital)
- **Risk:Reward**: 1:5 ratio

### Position Sizing:
- **Never exceed** 3 positions in LOW_VOL
- **Each position** should be ~$12,800
- **Total exposure** should be ~$38,400 (2.4x)

### Exit Rules:
- **Take profits** at 30% (don't get greedy)
- **Stop losses** at 6% (honor them!)
- **Time exit** after 14 days
- **Regime change** exits if market shifts

## 🚀 Quick Commands Reference

```bash
# Check current positions and P&L
python trading_dashboard.py

# Get take profit levels
python setup_take_profits.py

# Update system with actual positions
python fix_positions_now.py

# Run alerts once
python ml_adaptive_v49_alerts_fixed.py --once

# Start continuous monitoring
python ml_adaptive_v49_alerts_fixed.py
```

## ✅ Checklist for Tomorrow

- [ ] Wait for morning auto-check at 9:00 AM
- [ ] Review trading dashboard
- [ ] Execute AMD rebalancing (buy ~41 shares)
- [ ] Reduce TSLA if needed (sell ~5 shares)
- [ ] Set all take profit orders
- [ ] Verify all stop losses
- [ ] Update positions file after trades

---

**Remember**: The system achieved 155% CAGR by following these rules exactly. Stay disciplined! 🎯
