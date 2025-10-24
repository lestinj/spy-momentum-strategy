#!/usr/bin/env python3
"""
Analyze positions.txt vs script output
"""
import json
from datetime import datetime

def load_positions_txt(filename='positions.txt'):
    """Load positions from CSV file"""
    positions = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                symbol = parts[0]
                date = parts[1]
                entry_price = float(parts[2])
                shares = int(parts[3])
                
                positions[symbol] = {
                    'symbol': symbol,
                    'shares': shares,
                    'entry_price': entry_price,
                    'entry_date': date,
                    'capital_used': shares * entry_price
                }
    
    return positions

def analyze_discrepancy(total_capital=16000):
    """Analyze the discrepancy"""
    print("=" * 70)
    print("POSITION ANALYSIS - positions.txt vs Script Output")
    print("=" * 70)
    
    # Load positions.txt
    positions = load_positions_txt('positions.txt')
    
    print("\n📋 FROM positions.txt:")
    total_cost = 0
    for symbol, pos in positions.items():
        cost = pos['capital_used']
        total_cost += cost
        print(f"   {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${cost:,.2f}")
    
    print(f"\n   Total Cost: ${total_cost:,.2f}")
    
    # Script output shows
    print("\n📋 FROM Script Output:")
    script_positions = {
        'AMD': {'shares': 14, 'entry_price': 231.84},
        'TSLA': {'shares': 35, 'entry_price': 439.79},  # DISCREPANCY HERE
        'NVDA': {'shares': 65, 'entry_price': 183.66}
    }
    
    script_total = 0
    for symbol, pos in script_positions.items():
        cost = pos['shares'] * pos['entry_price']
        script_total += cost
        print(f"   {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${cost:,.2f}")
    
    print(f"\n   Total Cost: ${script_total:,.2f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("🔍 DISCREPANCY ANALYSIS:")
    print("=" * 70)
    
    print(f"\n{'Symbol':<8} {'positions.txt':<20} {'Script Output':<20} {'Match':<10}")
    print("-" * 70)
    
    for symbol in positions.keys():
        txt_shares = positions[symbol]['shares']
        txt_price = positions[symbol]['entry_price']
        
        if symbol in script_positions:
            script_shares = script_positions[symbol]['shares']
            script_price = script_positions[symbol]['entry_price']
            
            shares_match = "✅" if txt_shares == script_shares else "❌"
            price_match = "✅" if abs(txt_price - script_price) < 0.01 else "❌"
            
            print(f"{symbol:<8} {txt_shares} @ ${txt_price:.2f}{'':>6} {script_shares} @ ${script_price:.2f}{'':>6} {shares_match} {price_match}")
            
            if txt_shares != script_shares:
                diff = script_shares - txt_shares
                print(f"         ⚠️  SHARE DIFFERENCE: {diff:+d} shares")
            
            if abs(txt_price - script_price) >= 0.01:
                diff = script_price - txt_price
                print(f"         ⚠️  PRICE DIFFERENCE: ${diff:+.2f}")
    
    # Capital analysis
    print("\n" + "=" * 70)
    print("💰 CAPITAL ANALYSIS:")
    print("=" * 70)
    print(f"Total Capital:        ${total_capital:>12,.2f}")
    print(f"Cost (positions.txt): ${total_cost:>12,.2f}")
    print(f"Cost (script):        ${script_total:>12,.2f}")
    print(f"Available (txt):      ${(total_capital - total_cost):>12,.2f}")
    print(f"Available (script):   ${(total_capital - script_total):>12,.2f}")
    
    if total_cost > total_capital:
        print(f"\n⚠️  WARNING: Positions cost ${total_cost:,.2f} with only ${total_capital:,.2f} capital!")
        print(f"   You're over-leveraged by ${(total_cost - total_capital):,.2f}")
    
    if script_total > total_capital:
        print(f"\n⚠️  WARNING: Script positions cost ${script_total:,.2f} with only ${total_capital:,.2f} capital!")
        print(f"   Script shows over-leverage of ${(script_total - total_capital):,.2f}")
    
    # Root cause
    print("\n" + "=" * 70)
    print("🎯 ROOT CAUSE:")
    print("=" * 70)
    print("""
1. WRONG POSITION FILE: Script reads 'adaptive_positions.json', not 'positions.txt'
   
2. TSLA MISMATCH:
   - positions.txt shows: 27 shares @ $441.76
   - Script shows:        35 shares @ $439.79
   - Difference:          8 extra shares, $2.03 lower price
   
3. NEGATIVE CAPITAL: The script's positions cost $30,577 with only $16,000 capital
   - This suggests the script's position data is outdated or incorrect
   
RECOMMENDATIONS:
   a) Create 'adaptive_positions.json' from positions.txt (correct data)
   b) Verify actual brokerage positions match positions.txt
   c) Consider updating capital setting if you have more funds deployed
   d) Or close positions to match available capital
""")

if __name__ == '__main__':
    analyze_discrepancy()
