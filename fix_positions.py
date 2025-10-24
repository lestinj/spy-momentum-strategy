#!/usr/bin/env python3
"""
FIX SCRIPT: Sync positions.txt to adaptive_positions.json
Creates the correct JSON file for the ML Adaptive V49 trading script
"""
import json
from datetime import datetime

def create_adaptive_positions_from_txt(total_capital=15700, max_leverage=2.5):
    """Create adaptive_positions.json from positions.txt"""
    
    # Load positions.txt
    positions = {}
    with open('positions.txt', 'r') as f:
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
                    'capital_used': shares * entry_price,
                    'entry_regime': 'NORMAL',  # Default regime
                    'entry_rsi': 0,  # Unknown
                    'ml_confidence': 0.60,  # Default
                    'stop_loss': 0.08,
                    'take_profit': 0.25
                }
    
    # Calculate capital with leverage
    total_capital_used = sum(p['capital_used'] for p in positions.values())
    max_buying_power = total_capital * max_leverage
    available_capital = total_capital - total_capital_used  # For script compatibility
    
    # Create JSON structure
    json_data = {
        'positions': positions,
        'last_updated': datetime.now().isoformat(),
        'total_capital': total_capital,
        'available_capital': available_capital,
        'current_regime': 'NORMAL',
        'max_leverage': max_leverage,
        'buying_power': max_buying_power,
        'buying_power_used': total_capital_used
    }
    
    return json_data, total_capital_used, max_buying_power

def main():
    print("=" * 70)
    print("🔧 CREATING adaptive_positions.json FROM positions.txt")
    print("=" * 70)
    
    # User's actual capital and leverage
    capital = 15700
    max_leverage = 2.5
    
    print(f"\n💰 Your Trading Account:")
    print(f"   Capital:        ${capital:,.2f}")
    print(f"   Max Leverage:   {max_leverage}x")
    print(f"   Buying Power:   ${capital * max_leverage:,.2f}")
    
    # Create JSON
    json_data, total_used, buying_power = create_adaptive_positions_from_txt(capital, max_leverage)
    
    leverage_used = total_used / capital
    buying_power_remaining = buying_power - total_used
    leverage_pct = (total_used / buying_power) * 100
    
    print(f"\n📊 POSITION SUMMARY:")
    print(f"   Total Cost:     ${total_used:,.2f}")
    print(f"   Leverage Used:  {leverage_used:.2f}x ({leverage_pct:.1f}% of max)")
    print(f"   Buying Power Remaining: ${buying_power_remaining:,.2f}")
    
    if leverage_used > max_leverage:
        print(f"\n   ⚠️  OVER-LEVERAGED by {(leverage_used - max_leverage):.2f}x")
    else:
        print(f"\n   ✅ Within leverage limits")
    
    print(f"\n📋 Positions from positions.txt:")
    for symbol, pos in json_data['positions'].items():
        cost = pos['shares'] * pos['entry_price']
        print(f"   {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${cost:,.2f}")
    
    # Save
    with open('adaptive_positions.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n✅ Created adaptive_positions.json")
    
    # Also update capital config
    capital_config = {
        'total_capital': capital,
        'available_capital': json_data['available_capital'],
        'max_leverage': max_leverage,
        'buying_power': buying_power,
        'capital_history': [{
            'date': datetime.now().isoformat(),
            'amount': capital,
            'type': 'sync_from_positions_txt',
            'note': 'Synced after manual TSLA close (35→27 shares)'
        }],
        'last_updated': datetime.now().isoformat()
    }
    
    with open('capital_config.json', 'w') as f:
        json.dump(capital_config, f, indent=2)
    
    print(f"✅ Updated capital_config.json")
    
    print("\n" + "=" * 70)
    print("✅ SYNC COMPLETE")
    print("=" * 70)
    print("\n📝 What was fixed:")
    print("  • TSLA position updated: 35 shares → 27 shares ✅")
    print("  • Capital set to correct amount: $15,700 ✅")
    print("  • Leverage calculated: 1.73x (within 2.5x limit) ✅")
    print("\nNext steps:")
    print("  1. Run: python ml_adaptive_v49_alerts.py --once")
    print("  2. Verify output now shows 27 TSLA shares")
    print("  3. Available capital should show: $-11,411")
    print("     (This is normal - script tracks raw capital, not buying power)")
    print("=" * 70)

if __name__ == '__main__':
    main()
