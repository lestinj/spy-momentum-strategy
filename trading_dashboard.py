#!/usr/bin/env python3
"""
ML-ADAPTIVE V49 TRADING DASHBOARD
Complete position management and rebalancing tool
"""

import json
import yfinance as yf
from datetime import datetime
import sys

# Your positions
POSITIONS = {
    "AMD": {"shares": 14, "entry": 231.84, "date": "2025-10-17"},
    "TSLA": {"shares": 35, "entry": 439.79, "date": "2025-10-20"},  # Averaged
    "NVDA": {"shares": 65, "entry": 183.66, "date": "2025-10-20"}
}

# Strategy parameters
CAPITAL = 16000
TARGET_LEVERAGE = 2.4
MAX_POSITIONS = 3
TAKE_PROFIT = 0.30  # 30% in LOW_VOL
STOP_LOSS = 0.06    # 6% in LOW_VOL

# Calculate targets
TARGET_TOTAL = CAPITAL * TARGET_LEVERAGE  # $38,400
TARGET_PER_POSITION = TARGET_TOTAL / MAX_POSITIONS  # $12,800

def get_current_prices():
    """Get current prices for positions"""
    prices = {}
    for symbol in POSITIONS.keys():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                prices[symbol] = data['Close'].iloc[-1]
            else:
                prices[symbol] = POSITIONS[symbol]['entry']  # Fallback
        except:
            prices[symbol] = POSITIONS[symbol]['entry']
    return prices

def calculate_take_profit_orders():
    """Generate take profit orders"""
    orders = []
    for symbol, pos in POSITIONS.items():
        tp_price = pos['entry'] * (1 + TAKE_PROFIT)
        sl_price = pos['entry'] * (1 - STOP_LOSS)
        
        orders.append({
            'symbol': symbol,
            'action': 'TAKE_PROFIT',
            'shares': pos['shares'],
            'price': tp_price,
            'current_value': pos['shares'] * pos['entry'],
            'profit_if_hit': pos['shares'] * (tp_price - pos['entry']),
            'stop_loss': sl_price,
            'loss_if_stopped': pos['shares'] * (pos['entry'] - sl_price)
        })
    
    return orders

def calculate_rebalancing():
    """Calculate rebalancing recommendations"""
    rebalancing = []
    
    for symbol, pos in POSITIONS.items():
        current_value = pos['shares'] * pos['entry']
        target_value = TARGET_PER_POSITION
        difference = target_value - current_value
        
        if abs(difference) > 500:  # Only rebalance if >$500 difference
            shares_to_trade = int(difference / pos['entry'])
            rebalancing.append({
                'symbol': symbol,
                'current_shares': pos['shares'],
                'current_value': current_value,
                'target_value': target_value,
                'action': 'BUY' if shares_to_trade > 0 else 'SELL',
                'shares': abs(shares_to_trade),
                'new_total': pos['shares'] + shares_to_trade
            })
    
    return rebalancing

def print_dashboard():
    """Print the complete dashboard"""
    print("="*90)
    print("ML-ADAPTIVE V49 TRADING DASHBOARD")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*90)
    
    # Get current prices
    print("\n⏳ Fetching current prices...")
    current_prices = get_current_prices()
    
    # Current positions summary
    print("\n📊 CURRENT POSITIONS")
    print("-"*90)
    print(f"{'Symbol':<8} {'Shares':<8} {'Entry':<10} {'Current':<10} {'Value':<12} {'P&L':<12} {'P&L %':<8}")
    print("-"*90)
    
    total_value = 0
    total_cost = 0
    
    for symbol, pos in POSITIONS.items():
        current = current_prices.get(symbol, pos['entry'])
        cost = pos['shares'] * pos['entry']
        value = pos['shares'] * current
        pnl = value - cost
        pnl_pct = (pnl / cost) * 100
        
        total_value += value
        total_cost += cost
        
        # Color coding (terminal colors)
        color = '\033[92m' if pnl >= 0 else '\033[91m'  # Green or Red
        reset = '\033[0m'
        
        print(f"{symbol:<8} {pos['shares']:<8} ${pos['entry']:<9.2f} ${current:<9.2f} "
              f"${value:<11,.2f} {color}${pnl:<11,.2f} {pnl_pct:>7.1f}%{reset}")
    
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100
    leverage = total_cost / CAPITAL
    
    print("-"*90)
    print(f"{'TOTAL':<8} {'':<8} {'':<10} {'':<10} ${total_value:<11,.2f} "
          f"${total_pnl:<11,.2f} {total_pnl_pct:>7.1f}%")
    print(f"\nLeverage: {leverage:.2f}x of {TARGET_LEVERAGE}x target")
    print(f"Capital Usage: ${total_cost:,.2f} of ${TARGET_TOTAL:,.2f} capacity")
    
    # Take Profit Orders
    print("\n📈 TAKE PROFIT ORDERS (30% Target)")
    print("-"*90)
    print(f"{'Symbol':<8} {'Shares':<8} {'TP Price':<12} {'Current Gap':<12} {'Profit if Hit':<12}")
    print("-"*90)
    
    orders = calculate_take_profit_orders()
    total_potential = 0
    
    for order in orders:
        current = current_prices.get(order['symbol'], POSITIONS[order['symbol']]['entry'])
        gap_pct = ((order['price'] - current) / current) * 100
        total_potential += order['profit_if_hit']
        
        print(f"{order['symbol']:<8} {order['shares']:<8} ${order['price']:<11.2f} "
              f"{gap_pct:>10.1f}% away ${order['profit_if_hit']:>11,.2f}")
    
    print("-"*90)
    print(f"Total Potential Profit: ${total_potential:,.2f}")
    
    # Stop Loss Levels
    print("\n🛑 STOP LOSS LEVELS (6% Risk)")
    print("-"*90)
    print(f"{'Symbol':<8} {'Stop Price':<12} {'Current Gap':<12} {'Loss if Hit':<12}")
    print("-"*90)
    
    for order in orders:
        current = current_prices.get(order['symbol'], POSITIONS[order['symbol']]['entry'])
        gap_pct = ((current - order['stop_loss']) / current) * 100
        
        print(f"{order['symbol']:<8} ${order['stop_loss']:<11.2f} "
              f"{gap_pct:>10.1f}% away ${order['loss_if_stopped']:>11,.2f}")
    
    # Rebalancing Recommendations
    print("\n🔄 REBALANCING TO TARGET ($12,800 per position)")
    print("-"*90)
    
    rebalancing = calculate_rebalancing()
    
    if rebalancing:
        for rec in rebalancing:
            print(f"{rec['symbol']}: {rec['action']} {rec['shares']} shares")
            print(f"  Current: ${rec['current_value']:,.2f} → Target: ${rec['target_value']:,.2f}")
            print(f"  New position: {rec['new_total']} shares")
    else:
        print("✅ Positions are reasonably balanced")
    
    # Risk Metrics
    print("\n⚠️ RISK METRICS")
    print("-"*90)
    max_loss = sum(o['loss_if_stopped'] for o in orders)
    risk_pct = (max_loss / CAPITAL) * 100
    reward_ratio = total_potential / max_loss
    
    print(f"Maximum Loss (all stops hit): ${max_loss:,.2f} ({risk_pct:.1f}% of capital)")
    print(f"Risk:Reward Ratio: 1:{reward_ratio:.1f}")
    print(f"Break-even Win Rate: {(1/(1+reward_ratio))*100:.1f}%")
    
    # Action Items
    print("\n✅ ACTION ITEMS")
    print("-"*90)
    print("1. SET TAKE PROFIT ORDERS:")
    for order in orders:
        print(f"   {order['symbol']}: LIMIT SELL {order['shares']} @ ${order['price']:.2f} GTC")
    
    print("\n2. VERIFY STOP LOSSES:")
    for order in orders:
        print(f"   {order['symbol']}: STOP LOSS {order['shares']} @ ${order['stop_loss']:.2f}")
    
    if rebalancing:
        print("\n3. REBALANCING TRADES:")
        for rec in rebalancing:
            if rec['action'] == 'BUY':
                print(f"   BUY {rec['shares']} {rec['symbol']}")
            else:
                print(f"   SELL {rec['shares']} {rec['symbol']}")
    
    print("\n" + "="*90)
    
    # Save to file
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'positions': POSITIONS,
        'current_prices': {k: float(v) for k, v in current_prices.items()},
        'take_profit_orders': orders,
        'rebalancing': rebalancing,
        'metrics': {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'leverage': leverage,
            'max_loss': max_loss,
            'potential_profit': total_potential
        }
    }
    
    with open('trading_dashboard.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"📁 Dashboard saved to: trading_dashboard.json")
    print("="*90)

if __name__ == "__main__":
    print_dashboard()
    
    # Option to update positions
    print("\n💡 To update positions, edit POSITIONS dictionary in this script")
    print("   Then run: python trading_dashboard.py")
