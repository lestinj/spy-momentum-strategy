#!/usr/bin/env python3
"""
UNIFIED TRADING CONFIGURATION
==============================
Single source of truth for capital, positions, and settings
Used by BOTH live trading and backtesting scripts
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional


class TradingConfig:
    """
    Centralized configuration for trading operations
    Ensures consistency across live trading and backtesting
    """
    
    def __init__(self, config_file='trading_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            return self.create_default_config()
    
    def create_default_config(self) -> dict:
        """Create default configuration"""
        config = {
            'capital': {
                'total_capital': 15700,
                'max_leverage': 2.5,
                'buying_power': 39250.0,
                'last_updated': datetime.now().isoformat()
            },
            'positions': {},
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
                'version': '1.0'
            }
        }
        self.save_config(config)
        return config
    
    def save_config(self, config: Optional[dict] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        config['metadata']['last_saved'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Capital Management
    
    @property
    def total_capital(self) -> float:
        """Get total capital"""
        return self.config['capital']['total_capital']
    
    @total_capital.setter
    def total_capital(self, value: float):
        """Set total capital"""
        self.config['capital']['total_capital'] = value
        self.config['capital']['buying_power'] = value * self.max_leverage
        self.config['capital']['last_updated'] = datetime.now().isoformat()
        self.save_config()
    
    @property
    def max_leverage(self) -> float:
        """Get max leverage"""
        return self.config['capital']['max_leverage']
    
    @max_leverage.setter
    def max_leverage(self, value: float):
        """Set max leverage"""
        self.config['capital']['max_leverage'] = value
        self.config['capital']['buying_power'] = self.total_capital * value
        self.save_config()
    
    @property
    def buying_power(self) -> float:
        """Get total buying power"""
        return self.config['capital']['buying_power']
    
    @property
    def available_capital(self) -> float:
        """Calculate available capital after positions"""
        total_in_positions = sum(
            p.get('shares', 0) * p.get('entry_price', 0) 
            for p in self.positions.values()
        )
        return self.total_capital - total_in_positions
    
    @property
    def available_buying_power(self) -> float:
        """Calculate available buying power after positions"""
        total_in_positions = sum(
            p.get('shares', 0) * p.get('entry_price', 0) 
            for p in self.positions.values()
        )
        return self.buying_power - total_in_positions
    
    @property
    def current_leverage(self) -> float:
        """Calculate current leverage"""
        total_in_positions = sum(
            p.get('shares', 0) * p.get('entry_price', 0) 
            for p in self.positions.values()
        )
        return total_in_positions / self.total_capital if self.total_capital > 0 else 0
    
    # Position Management
    
    @property
    def positions(self) -> Dict:
        """Get current positions"""
        return self.config['positions']
    
    def add_position(self, symbol: str, shares: int, entry_price: float, 
                     entry_date: str, **kwargs):
        """Add a position"""
        self.config['positions'][symbol] = {
            'symbol': symbol,
            'shares': shares,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'capital_used': shares * entry_price,
            **kwargs
        }
        self.save_config()
    
    def remove_position(self, symbol: str):
        """Remove a position"""
        if symbol in self.config['positions']:
            del self.config['positions'][symbol]
            self.save_config()
    
    def update_position(self, symbol: str, **kwargs):
        """Update position attributes"""
        if symbol in self.config['positions']:
            self.config['positions'][symbol].update(kwargs)
            self.save_config()
    
    def load_positions_from_txt(self, filename='positions.txt'):
        """Load positions from CSV file"""
        if not os.path.exists(filename):
            print(f"⚠️  {filename} not found")
            return
        
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
                        'capital_used': shares * entry_price,
                        'entry_regime': 'NORMAL',
                        'stop_loss': self.config['risk']['base_stop_loss'],
                        'take_profit': self.config['risk']['base_take_profit']
                    }
        
        self.config['positions'] = positions
        self.save_config()
        print(f"✅ Loaded {len(positions)} positions from {filename}")
    
    # Risk Parameters
    
    @property
    def base_stop_loss(self) -> float:
        return self.config['risk']['base_stop_loss']
    
    @property
    def base_take_profit(self) -> float:
        return self.config['risk']['base_take_profit']
    
    @property
    def max_positions(self) -> int:
        return self.config['risk']['max_positions']
    
    # Strategy Parameters
    
    @property
    def symbols(self) -> list:
        return self.config['strategy']['symbols']
    
    @property
    def rsi_period(self) -> int:
        return self.config['strategy']['rsi_period']
    
    @property
    def rsi_buy(self) -> int:
        return self.config['strategy']['rsi_buy']
    
    @property
    def rsi_sell(self) -> int:
        return self.config['strategy']['rsi_sell']
    
    @property
    def ma_fast(self) -> int:
        return self.config['strategy']['ma_fast']
    
    @property
    def ma_slow(self) -> int:
        return self.config['strategy']['ma_slow']
    
    # ML Parameters
    
    @property
    def min_ml_accuracy(self) -> float:
        return self.config['ml']['min_accuracy']
    
    @property
    def high_confidence(self) -> float:
        return self.config['ml']['high_confidence']
    
    @property
    def low_confidence(self) -> float:
        return self.config['ml']['low_confidence']
    
    # Capital Operations
    
    def add_capital(self, amount: float, note: str = ''):
        """Add capital to the account"""
        old_capital = self.total_capital
        self.total_capital = old_capital + amount
        
        if 'capital_history' not in self.config['capital']:
            self.config['capital']['capital_history'] = []
        
        self.config['capital']['capital_history'].append({
            'date': datetime.now().isoformat(),
            'type': 'deposit',
            'amount': amount,
            'old_total': old_capital,
            'new_total': self.total_capital,
            'note': note
        })
        
        self.save_config()
        print(f"✅ Added ${amount:,.2f} → Total: ${self.total_capital:,.2f}")
    
    def withdraw_capital(self, amount: float, note: str = ''):
        """Withdraw capital from the account"""
        old_capital = self.total_capital
        self.total_capital = old_capital - amount
        
        if 'capital_history' not in self.config['capital']:
            self.config['capital']['capital_history'] = []
        
        self.config['capital']['capital_history'].append({
            'date': datetime.now().isoformat(),
            'type': 'withdrawal',
            'amount': -amount,
            'old_total': old_capital,
            'new_total': self.total_capital,
            'note': note
        })
        
        self.save_config()
        print(f"✅ Withdrew ${amount:,.2f} → Total: ${self.total_capital:,.2f}")
    
    # Display Methods
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print("📊 TRADING CONFIGURATION SUMMARY")
        print("=" * 70)
        
        print(f"\n💰 Capital:")
        print(f"   Total Capital:       ${self.total_capital:,.2f}")
        print(f"   Max Leverage:        {self.max_leverage}x")
        print(f"   Total Buying Power:  ${self.buying_power:,.2f}")
        
        if self.positions:
            total_in_positions = sum(p['capital_used'] for p in self.positions.values())
            print(f"\n📋 Current Positions ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                print(f"   {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${pos['capital_used']:,.2f}")
            print(f"   Total in Positions:  ${total_in_positions:,.2f}")
            print(f"   Current Leverage:    {self.current_leverage:.2f}x")
            print(f"   Available Capital:   ${self.available_capital:,.2f}")
            print(f"   Available Power:     ${self.available_buying_power:,.2f}")
        else:
            print(f"\n📋 Positions: None")
        
        print(f"\n🎯 Risk Parameters:")
        print(f"   Stop Loss:           {self.base_stop_loss*100:.0f}%")
        print(f"   Take Profit:         {self.base_take_profit*100:.0f}%")
        print(f"   Max Positions:       {self.max_positions}")
        
        print(f"\n📈 Strategy:")
        print(f"   Symbols:             {', '.join(self.symbols)}")
        print(f"   RSI Buy/Sell:        {self.rsi_buy}/{self.rsi_sell}")
        
        print("=" * 70)


# Command Line Interface
if __name__ == "__main__":
    import sys
    
    config = TradingConfig()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'show':
            config.print_summary()
        
        elif command == 'set-capital':
            if len(sys.argv) > 2:
                amount = float(sys.argv[2])
                config.total_capital = amount
                print(f"✅ Capital set to ${amount:,.2f}")
            else:
                print("Usage: python trading_config.py set-capital <amount>")
        
        elif command == 'add-capital':
            if len(sys.argv) > 2:
                amount = float(sys.argv[2])
                note = ' '.join(sys.argv[3:]) if len(sys.argv) > 3 else ''
                config.add_capital(amount, note)
            else:
                print("Usage: python trading_config.py add-capital <amount> [note]")
        
        elif command == 'set-leverage':
            if len(sys.argv) > 2:
                leverage = float(sys.argv[2])
                config.max_leverage = leverage
                print(f"✅ Leverage set to {leverage}x")
            else:
                print("Usage: python trading_config.py set-leverage <multiplier>")
        
        elif command == 'load-positions':
            filename = sys.argv[2] if len(sys.argv) > 2 else 'positions.txt'
            config.load_positions_from_txt(filename)
            config.print_summary()
        
        else:
            print("Unknown command. Available commands:")
            print("  show              - Display current configuration")
            print("  set-capital       - Set total capital")
            print("  add-capital       - Add capital to account")
            print("  set-leverage      - Set max leverage")
            print("  load-positions    - Load positions from positions.txt")
    else:
        config.print_summary()
