#!/usr/bin/env python3
"""
TAKE PROFIT CALCULATOR & ORDER GENERATOR
For ML-Adaptive V49 Strategy
Calculates exact take profit levels for your positions
"""

import json
from datetime import datetime

# Your current positions
positions = {
    "AMD": {
        "shares": 14,
        "entry_price": 231.84,
        "entry_date": "2025-10-17"
    },
    "TSLA": {
        "shares": 35,  # Combined 8 + 27
        "avg_entry": 439.79,  # Weighted average
        "entry_dates": ["2025-10-17", "2025-10-20"],
        "breakdown": "8 @ 435.30 + 27 @ 441.76"
    },
    "NVDA": {
        "shares": 65,
        "entry_price": 183.66,
        "entry_date": "2025-10-20"
    }
}

# Strategy parameters in LOW_VOL regime
TAKE_PROFIT_PCT = 0.30  # 30% gain target
STOP_LOSS_PCT = 0.06    # 6% stop loss

print("="*80)
print("ML-ADAPTIVE V49 - TAKE PROFIT ORDERS")
print("="*80)
print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Regime: LOW_VOL (30% take profit, 6% stop loss)")
print("="*80)

print("\n📈 TAKE PROFIT ORDERS TO SET:\n")

# Calculate take profits
total_potential_profit = 0

for symbol in ["AMD", "TSLA", "NVDA"]:
    if symbol == "TSLA":
        entry = positions[symbol]["avg_entry"]
    else:
        entry = positions[symbol]["entry_price"]
    
    shares = positions[symbol]["shares"]
    
    # Calculate levels
    take_profit_price = entry * (1 + TAKE_PROFIT_PCT)
    stop_loss_price = entry * (1 - STOP_LOSS_PCT)
    
    # Calculate dollar amounts
    current_value = shares * entry
    profit_if_hit = shares * (take_profit_price - entry)
    loss_if_stopped = shares * (entry - stop_loss_price)
    
    print(f"{'='*80}")
    print(f"🎯 {symbol}")
    print(f"{'='*80}")
    print(f"Position: {shares} shares @ ${entry:.2f} = ${current_value:,.2f}")
    print()
    print(f"📈 TAKE PROFIT ORDER:")
    print(f"   Type: LIMIT SELL")
    print(f"   Shares: {shares}")
    print(f"   Price: ${take_profit_price:.2f}")
    print(f"   Good Till: GTC (Good Till Cancelled)")
    print(f"   Expected Profit: ${profit_if_hit:,.2f} (+30%)")
    print()
    print(f"🛑 STOP LOSS ORDER (verify you have this):")
    print(f"   Type: STOP LOSS")
    print(f"   Shares: {shares}")
    print(f"   Stop Price: ${stop_loss_price:.2f}")
    print(f"   Max Loss: ${loss_if_stopped:,.2f} (-6%)")
    print()
    
    total_potential_profit += profit_if_hit

# Generate order entry templates
print(f"{'='*80}")
print("📋 BROKER ORDER ENTRY TEMPLATE")
print(f"{'='*80}")
print("\nCopy these into your broker platform:\n")

for symbol in ["AMD", "TSLA", "NVDA"]:
    if symbol == "TSLA":
        entry = positions[symbol]["avg_entry"]
    else:
        entry = positions[symbol]["entry_price"]
    
    shares = positions[symbol]["shares"]
    take_profit_price = entry * (1 + TAKE_PROFIT_PCT)
    
    print(f"{symbol}: SELL {shares} @ LIMIT ${take_profit_price:.2f} GTC")

print(f"\n{'='*80}")
print("📊 SUMMARY")
print(f"{'='*80}")
print(f"Total Invested: ${sum(p['shares'] * (p.get('entry_price', p.get('avg_entry'))) for p in positions.values()):,.2f}")
print(f"Total Potential Profit (if all TP hit): ${total_potential_profit:,.2f}")
print(f"Risk:Reward Ratio: 1:5 (6% risk for 30% reward)")

print(f"\n💡 IMPORTANT NOTES:")
print(f"   1. Set these as GTC (Good Till Cancelled) orders")
print(f"   2. These will auto-execute when price hits target")
print(f"   3. You can adjust if market regime changes")
print(f"   4. Consider trailing stops after 20% gain")

# Save to file for reference
orders_data = {}
for symbol in ["AMD", "TSLA", "NVDA"]:
    if symbol == "TSLA":
        entry = positions[symbol]["avg_entry"]
    else:
        entry = positions[symbol]["entry_price"]
    
    shares = positions[symbol]["shares"]
    
    orders_data[symbol] = {
        "shares": shares,
        "entry_price": entry,
        "take_profit_price": round(entry * (1 + TAKE_PROFIT_PCT), 2),
        "stop_loss_price": round(entry * (1 - STOP_LOSS_PCT), 2),
        "potential_profit": round(shares * (entry * TAKE_PROFIT_PCT), 2),
        "max_loss": round(shares * (entry * STOP_LOSS_PCT), 2)
    }

with open('take_profit_orders.json', 'w') as f:
    json.dump({
        "generated": datetime.now().isoformat(),
        "regime": "LOW_VOL",
        "take_profit_pct": TAKE_PROFIT_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "orders": orders_data
    }, f, indent=2)

print(f"\n✅ Orders saved to: take_profit_orders.json")
print(f"{'='*80}")
