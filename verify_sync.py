#!/usr/bin/env python3
"""
Verify that positions are correctly synced
Run this AFTER fix_positions.py
"""
import json
import os

def verify_sync():
    """Verify positions.txt matches adaptive_positions.json"""
    
    print("=" * 70)
    print("🔍 VERIFICATION: Checking if sync was successful")
    print("=" * 70)
    
    # Check files exist
    if not os.path.exists('positions.txt'):
        print("❌ positions.txt not found!")
        return False
    
    if not os.path.exists('adaptive_positions.json'):
        print("❌ adaptive_positions.json not found!")
        print("   Run fix_positions.py first!")
        return False
    
    # Load positions.txt
    txt_positions = {}
    with open('positions.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                symbol = parts[0]
                txt_positions[symbol] = {
                    'shares': int(parts[3]),
                    'entry_price': float(parts[2]),
                    'date': parts[1]
                }
    
    # Load adaptive_positions.json
    with open('adaptive_positions.json', 'r') as f:
        json_data = json.load(f)
        json_positions = json_data.get('positions', {})
    
    # Compare
    print("\n📊 COMPARISON:")
    print(f"\n{'Symbol':<8} {'positions.txt':<25} {'adaptive_positions.json':<25} {'Status':<10}")
    print("-" * 70)
    
    all_match = True
    
    for symbol in txt_positions:
        txt = txt_positions[symbol]
        
        if symbol in json_positions:
            json_pos = json_positions[symbol]
            
            shares_match = txt['shares'] == json_pos['shares']
            price_match = abs(txt['entry_price'] - json_pos['entry_price']) < 0.01
            
            txt_str = f"{txt['shares']} @ ${txt['entry_price']:.2f}"
            json_str = f"{json_pos['shares']} @ ${json_pos['entry_price']:.2f}"
            
            if shares_match and price_match:
                status = "✅ MATCH"
            else:
                status = "❌ DIFFER"
                all_match = False
            
            print(f"{symbol:<8} {txt_str:<25} {json_str:<25} {status:<10}")
            
            if not shares_match:
                print(f"         ⚠️  Share mismatch: {txt['shares']} vs {json_pos['shares']}")
            if not price_match:
                print(f"         ⚠️  Price mismatch: ${txt['entry_price']:.2f} vs ${json_pos['entry_price']:.2f}")
        else:
            print(f"{symbol:<8} {txt['shares']} shares        NOT IN JSON           ❌ MISSING")
            all_match = False
    
    # Check for extra positions in JSON
    for symbol in json_positions:
        if symbol not in txt_positions:
            json_pos = json_positions[symbol]
            json_str = f"{json_pos['shares']} @ ${json_pos['entry_price']:.2f}"
            print(f"{symbol:<8} NOT IN TXT              {json_str:<25} ❌ EXTRA")
            all_match = False
    
    # Capital check
    print("\n" + "=" * 70)
    print("💰 CAPITAL VERIFICATION:")
    print("=" * 70)
    
    expected_capital = 15700
    expected_leverage = 2.5
    
    actual_capital = json_data.get('total_capital', 0)
    max_leverage = json_data.get('max_leverage', 2.0)
    
    print(f"Expected Capital:  ${expected_capital:,.2f}")
    print(f"Actual Capital:    ${actual_capital:,.2f}  {'✅' if actual_capital == expected_capital else '❌'}")
    print(f"\nExpected Leverage: {expected_leverage}x")
    print(f"Actual Leverage:   {max_leverage}x  {'✅' if max_leverage == expected_leverage else '❌'}")
    
    # Calculate position values
    total_cost = sum(p['shares'] * p['entry_price'] for p in json_positions.values())
    leverage_used = total_cost / actual_capital if actual_capital > 0 else 0
    buying_power = actual_capital * max_leverage
    
    print(f"\nPosition Total:    ${total_cost:,.2f}")
    print(f"Leverage Used:     {leverage_used:.2f}x")
    print(f"Buying Power:      ${buying_power:,.2f}")
    print(f"Remaining Power:   ${buying_power - total_cost:,.2f}")
    
    if leverage_used > max_leverage:
        print(f"\n⚠️  WARNING: Over-leveraged by {(leverage_used - max_leverage):.2f}x")
        all_match = False
    else:
        print(f"\n✅ Within leverage limits")
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_match:
        print("✅ ✅ ✅ ALL CHECKS PASSED! ✅ ✅ ✅")
        print("=" * 70)
        print("\nYour positions are perfectly synced!")
        print("You can now run: python ml_adaptive_v49_alerts.py --once")
        print("\nExpected output:")
        print("  📋 Current Positions:")
        for symbol, pos in txt_positions.items():
            print(f"     {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
    else:
        print("❌ SYNC ISSUES DETECTED")
        print("=" * 70)
        print("\nSomething didn't sync correctly.")
        print("Try running fix_positions.py again.")
    
    print("=" * 70)
    
    return all_match

if __name__ == '__main__':
    verify_sync()
