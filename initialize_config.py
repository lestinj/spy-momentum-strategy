#!/usr/bin/env python3
"""
INITIALIZE UNIFIED TRADING CONFIG
==================================
One-time setup to create trading_config.json from your current positions
"""

import json
from datetime import datetime
import os


def initialize_config_from_positions():
    """Initialize trading config from positions.txt"""
    
    print("=" * 70)
    print("🔧 INITIALIZING UNIFIED TRADING CONFIGURATION")
    print("=" * 70)
    
    # Load positions from positions.txt
    positions = {}
    if os.path.exists('positions.txt'):
        print("\n📋 Loading positions from positions.txt...")
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
                        'entry_regime': 'NORMAL',
                        'stop_loss': 0.08,
                        'take_profit': 0.25
                    }
        
        print(f"   ✅ Loaded {len(positions)} positions:")
        for symbol, pos in positions.items():
            print(f"      {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
    else:
        print("\n⚠️  positions.txt not found - starting with no positions")
    
    # Your actual capital and leverage
    total_capital = 15700
    max_leverage = 2.5
    
    # Create unified config
    config = {
        'capital': {
            'total_capital': total_capital,
            'max_leverage': max_leverage,
            'buying_power': total_capital * max_leverage,
            'last_updated': datetime.now().isoformat(),
            'capital_history': [{
                'date': datetime.now().isoformat(),
                'type': 'initial_setup',
                'amount': total_capital,
                'note': 'Unified config initialization'
            }]
        },
        'positions': positions,
        'risk': {
            'base_stop_loss': 0.08,
            'base_take_profit': 0.25,
            'max_positions': 3
        },
        'strategy': {
            'symbols': ['NVDA', 'TSLA', 'PLTR', 'AMD', 'COIN', 'META', 'NET'],
            'rsi_period': 14,
            'rsi_buy': 55,
            'rsi_sell': 45,
            'ma_fast': 10,
            'ma_slow': 30
        },
        'ml': {
            'min_accuracy': 0.45,
            'high_confidence': 0.60,
            'low_confidence': 0.30
        },
        'metadata': {
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'Unified configuration for live trading and backtesting'
        }
    }
    
    # Save config
    with open('trading_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Created trading_config.json")
    
    # Calculate metrics
    total_in_positions = sum(p['capital_used'] for p in positions.values())
    current_leverage = total_in_positions / total_capital if total_capital > 0 else 0
    available_buying_power = (total_capital * max_leverage) - total_in_positions
    
    print("\n" + "=" * 70)
    print("📊 CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\n💰 Capital:")
    print(f"   Total Capital:       ${total_capital:,.2f}")
    print(f"   Max Leverage:        {max_leverage}x")
    print(f"   Total Buying Power:  ${total_capital * max_leverage:,.2f}")
    
    if positions:
        print(f"\n📋 Positions ({len(positions)}):")
        for symbol, pos in positions.items():
            print(f"   {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${pos['capital_used']:,.2f}")
        print(f"\n   Total in Positions:  ${total_in_positions:,.2f}")
        print(f"   Current Leverage:    {current_leverage:.2f}x")
        print(f"   Available Power:     ${available_buying_power:,.2f}")
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Both scripts now use trading_config.json as single source of truth")
    print("  2. Update capital: python trading_config.py add-capital 5000")
    print("  3. View config:    python trading_config.py show")
    print("  4. Load positions: python trading_config.py load-positions")
    print("=" * 70)


if __name__ == '__main__':
    initialize_config_from_positions()
