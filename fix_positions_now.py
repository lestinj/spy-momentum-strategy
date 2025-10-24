#!/usr/bin/env python3
"""
CRITICAL: Update ML-Adaptive system with your ACTUAL positions
This will prevent double-buying!
"""

import json
from datetime import datetime

# YOUR ACTUAL POSITIONS
positions_data = {
    "positions": {
        "AMD": {
            "entry_price": 231.84,
            "entry_date": "2025-10-17",
            "shares": 14,
            "capital_used": 3245.76,  # 14 * 231.84
            "entry_value": 3245.76,
            "leveraged_value": 3245.76,
            "stop_loss": 0.06,
            "take_profit": 0.30,
            "ml_confidence": 0.45,
            "position_size": 0.20,
            "regime": "LOW_VOL"
        },
        "TSLA": {
            "entry_price": 439.79,  # Average of your two entries
            "entry_date": "2025-10-20",
            "shares": 35,  # Combined: 8 + 27
            "capital_used": 15392.65,  # 35 * 439.79
            "entry_value": 15392.65,
            "leveraged_value": 15392.65,
            "stop_loss": 0.06,
            "take_profit": 0.30,
            "ml_confidence": 0.45,
            "position_size": 0.96,  # This is huge!
            "regime": "LOW_VOL"
        },
        "NVDA": {
            "entry_price": 183.66,
            "entry_date": "2025-10-20",
            "shares": 65,
            "capital_used": 11937.90,  # 65 * 183.66
            "entry_value": 11937.90,
            "leveraged_value": 11937.90,
            "stop_loss": 0.06,
            "take_profit": 0.30,
            "ml_confidence": 0.42,
            "position_size": 0.75,  # This is huge!
            "regime": "LOW_VOL"
        }
    },
    "last_updated": datetime.now().isoformat(),
    "total_capital": 16000,
    "available_capital": -14576.31,  # 16000 - 30576.31 (NEGATIVE!)
    "current_regime": "LOW_VOL"
}

# Calculate actual totals
total_used = sum(p['capital_used'] for p in positions_data['positions'].values())
actual_available = 16000 - total_used

print("="*70)
print("UPDATING SYSTEM WITH YOUR ACTUAL POSITIONS")
print("="*70)
print(f"\n📊 YOUR REAL SITUATION:")
print(f"   Total Capital:    $16,000")
print(f"   Already Invested: ${total_used:,.2f}")
print(f"   Available:        ${actual_available:,.2f} ⚠️")
print(f"   Leverage Used:    {total_used/16000:.2f}x")
print(f"   Position Count:   {len(positions_data['positions'])}/3 max")

print(f"\n📋 CURRENT POSITIONS:")
for symbol, pos in positions_data['positions'].items():
    value = pos['shares'] * pos['entry_price']
    pct_of_capital = (value / 16000) * 100
    print(f"   {symbol:5} {pos['shares']:3} shares @ ${pos['entry_price']:6.2f} = ${value:8,.2f} ({pct_of_capital:5.1f}% of capital)")

# Update the available capital correctly
positions_data['available_capital'] = actual_available

# Save the CORRECT positions file
with open('adaptive_positions.json', 'w') as f:
    json.dump(positions_data, f, indent=2)

print(f"\n✅ POSITIONS FILE UPDATED!")
print(f"\n⚠️  WARNINGS:")
print(f"   1. You're using {total_used/16000:.1%} of capital (191%)")
print(f"   2. You have 3 positions (at max already)")
print(f"   3. Available capital is NEGATIVE: ${actual_available:,.2f}")
print(f"   4. System will now CORRECTLY show no buy signals")

print(f"\n🛑 CRITICAL ACTIONS:")
print(f"   1. STOP the current alerts program (Ctrl+C)")
print(f"   2. Run this script to update positions")
print(f"   3. Restart alerts - it should show NO available slots")

print(f"\n📊 STOP LOSSES (protect yourself!):")
for symbol, pos in positions_data['positions'].items():
    stop_price = pos['entry_price'] * (1 - pos['stop_loss'])
    current_loss_if_hit = pos['shares'] * (pos['entry_price'] - stop_price)
    print(f"   {symbol}: ${stop_price:.2f} (would lose ${current_loss_if_hit:,.2f})")

print(f"\n📈 TAKE PROFIT TARGETS:")
for symbol, pos in positions_data['positions'].items():
    target_price = pos['entry_price'] * (1 + pos['take_profit'])
    potential_gain = pos['shares'] * (target_price - pos['entry_price'])
    print(f"   {symbol}: ${target_price:.2f} (would gain ${potential_gain:,.2f})")

print("="*70)
print("⚠️  RUN THIS SCRIPT NOW TO FIX THE SYSTEM!")
print("="*70)
