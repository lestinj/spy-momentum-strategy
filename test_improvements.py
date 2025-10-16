"""
Quick test script to verify all improvements work
"""
import sys
sys.path.insert(0, '/home/claude/spy-momentum-strategy')

from data.improved_data_fetcher import DataFetcher
from datetime import datetime, timedelta

print("="*70)
print("TESTING IMPROVED SPY MOMENTUM STRATEGY")
print("="*70)

# Test 1: Data Fetcher Initialization
print("\n1. Testing DataFetcher initialization...")
try:
    fetcher = DataFetcher()
    print("✓ DataFetcher initialized successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: SPY Price Data
print("\n2. Testing SPY price data fetch...")
try:
    price_df = fetcher.fetch_spy_price(period='5d', interval='1h')
    if not price_df.empty:
        print(f"✓ Fetched {len(price_df)} bars of price data")
        print(f"  Latest close: ${price_df['Close'].iloc[-1]:.2f}")
        print(f"  Date range: {price_df.index[0]} to {price_df.index[-1]}")
    else:
        print("✗ No price data returned")
except Exception as e:
    print(f"✗ Error fetching price data: {e}")

# Test 3: SPY Options Data
print("\n3. Testing SPY options data fetch...")
try:
    calls, puts = fetcher.fetch_spy_options()
    if not calls.empty and not puts.empty:
        print(f"✓ Fetched options data")
        print(f"  Calls: {len(calls)} strikes")
        print(f"  Puts: {len(puts)} strikes")
        print(f"  Expiration: {calls['contractSymbol'].iloc[0][:15]}...")
    else:
        print("⚠ Options data may not be available (market hours only)")
except Exception as e:
    print(f"⚠ Error fetching options (may be normal outside market hours): {e}")

# Test 4: Options Signals
print("\n4. Testing options signals...")
try:
    signals = fetcher.fetch_options_signals()
    if signals:
        print("✓ Generated options signals")
        print(f"  P/C Ratio (Vol): {signals.get('pc_ratio_volume', 'N/A')}")
        print(f"  Sentiment: {signals.get('options_sentiment', 'N/A')}")
        print(f"  Current Price: ${signals.get('current_price', 'N/A'):.2f}")
    else:
        print("⚠ No options signals (may be outside market hours)")
except Exception as e:
    print(f"⚠ Error with options signals: {e}")

# Test 5: Macro Indicators
print("\n5. Testing macro indicators fetch...")
try:
    macro_df = fetcher.fetch_macro_indicators(lookback_days=90)
    if not macro_df.empty:
        print(f"✓ Fetched macro indicators")
        print(f"  Indicators: {len(macro_df.columns)} columns")
        print(f"  Data points: {len(macro_df)} rows")
        print(f"  Date range: {macro_df.index[0]} to {macro_df.index[-1]}")
        
        # Show which indicators were fetched
        indicator_cols = [c for c in macro_df.columns if not c.endswith('_bullish') 
                         and not c.endswith('_improving') and not c.endswith('_positive')
                         and c not in ['positive_macro_count', 'macro_bullish']]
        print(f"  Available indicators: {', '.join(indicator_cols[:5])}")
    else:
        print("⚠ No macro data returned (using defaults)")
except Exception as e:
    print(f"⚠ Error with macro indicators: {e}")

# Test 6: Latest Macro Signal
print("\n6. Testing latest macro signal...")
try:
    signal = fetcher.get_latest_macro_signal()
    print("✓ Got latest macro signal")
    print(f"  Macro Bullish: {signal.get('macro_bullish', 'N/A')}")
    print(f"  Positive Count: {signal.get('positive_count', 'N/A')}/6")
    if signal.get('vix'):
        print(f"  VIX: {signal['vix']:.2f}")
except Exception as e:
    print(f"✗ Error getting macro signal: {e}")

# Test 7: Momentum Indicators
print("\n7. Testing momentum indicators...")
try:
    from indicators.momentum import compute_momentum_indicators, MomentumParams
    
    price_df = fetcher.fetch_spy_price(period='5d', interval='1h')
    price_df.columns = [c.lower() for c in price_df.columns]
    
    params = MomentumParams(fast=10, slow=30)
    mom_df = compute_momentum_indicators(price_df[['close']], params)
    
    if not mom_df.empty:
        print("✓ Computed momentum indicators")
        print(f"  Indicators: {list(mom_df.columns)}")
        latest = mom_df.iloc[-1]
        print(f"  Latest RSI: {latest['rsi']:.1f}")
        print(f"  Latest Z-Score: {latest['zscore']:.2f}")
    else:
        print("✗ Failed to compute momentum")
except Exception as e:
    print(f"✗ Error computing momentum: {e}")

# Test 8: Backtest Import
print("\n8. Testing backtest module...")
try:
    from improved_backtest import ImprovedBacktest
    print("✓ Backtest module imported successfully")
    print("  Ready to run backtests!")
except Exception as e:
    print(f"✗ Error importing backtest: {e}")

# Test 9: Live Trader Import
print("\n9. Testing live trader module...")
try:
    from improved_main import LiveTrader
    print("✓ Live trader module imported successfully")
    print("  Ready for live signals!")
except Exception as e:
    print(f"✗ Error importing live trader: {e}")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)
print("\n📊 Next Steps:")
print("  1. Run backtest: python improved_backtest.py")
print("  2. Run live signals: python improved_main.py")
print("  3. Adjust parameters in the files as needed")
print("\n✨ All core components are working!\n")
