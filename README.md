# 🔧 FIX YOUR TRADING SCRIPT - Quick Start

## The Problem
Your script shows **35 TSLA shares** but you actually have **27 shares**.

## The Solution (2 Minutes)
Download these two files and place them next to your trading script:

---

## 📥 DOWNLOAD THESE FILES:

### 1. [adaptive_positions.json](computer:///mnt/user-data/outputs/adaptive_positions.json)
**Your correct position data**
- AMD: 14 shares @ $231.84 ✅
- NVDA: 65 shares @ $183.66 ✅
- TSLA: 27 shares @ $441.76 ✅ (CORRECTED!)

### 2. [capital_config.json](computer:///mnt/user-data/outputs/capital_config.json)
**Your capital settings**
- Capital: $15,700 ✅
- Max Leverage: 2.5x ✅
- Buying Power: $39,250 ✅

---

## 📍 WHERE TO PUT THEM:

Place both files in the **SAME FOLDER** as `ml_adaptive_v49_alerts.py`

```
📁 Your Trading Folder/
   📄 ml_adaptive_v49_alerts.py
   📄 positions.txt
   📄 adaptive_positions.json      ← PUT HERE
   📄 capital_config.json          ← PUT HERE
```

---

## ✅ VERIFY IT WORKED:

Run your script:
```bash
python ml_adaptive_v49_alerts.py --once
```

**You should now see:**
```
Capital: $15,700                 ✅ (was $16,000)
📋 Current Positions:
   TSLA: 27 shares @ $441.76     ✅ (was 35 @ $439.79)
```

---

## 📚 Additional Resources:

- [Installation Instructions](computer:///mnt/user-data/outputs/INSTALLATION_INSTRUCTIONS.txt) - Detailed walkthrough
- [Complete Fix Guide](computer:///mnt/user-data/outputs/COMPLETE_FIX_GUIDE.md) - Full documentation
- [fix_positions.py](computer:///mnt/user-data/outputs/fix_positions.py) - Auto-generate these files from positions.txt

---

## 💰 Your Position Status:

| Metric | Value |
|--------|-------|
| Capital | $15,700 |
| Positions Cost | $27,111 |
| Leverage Used | 1.73x |
| Max Leverage | 2.5x |
| Status | ✅ Healthy (69% used) |
| Buying Power Remaining | $12,139 |

---

## ❓ Quick Help:

**Q: Script still shows wrong data?**  
A: Files not in the same directory as your script

**Q: Can't find my script's directory?**  
A: Look where you run the script from

**Q: Files have wrong extension?**  
A: Make sure they end in `.json` not `.json.txt`

---

**That's it!** Download the two JSON files, drop them next to your script, and run it. ✨
