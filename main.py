import time
from datetime import datetime, timedelta
import pandas as pd

from data.yahoo_data import fetch_spy_price_data
from data.macro_data import fetch_macro_indicator
from indicators.momentum import price_momentum
from indicators.macro_momentum import macro_momentum
from indicators.alignment_checker import is_aligned
from strategy.trade_signal import generate_trade_signal

def run_live_loop(interval_minutes=5):
    while True:
        print(f"\n--- {datetime.now()} ---")

        # 1. Fetch SPY price data (past 5 days, 5m interval)
        price_df = fetch_spy_price_data(period="5d", interval="5m")
        if price_df.empty:
            print("Failed to fetch SPY price data.")
            time.sleep(interval_minutes * 60)
            continue

        # 2. Get macro indicator (e.g., Leading Economic Index)
        today = datetime.today()
        macro_df = fetch_macro_indicator("USSLIND", start=today - timedelta(days=180), end=today)
        if macro_df.empty:
            print("Failed to fetch macro indicator.")
            time.sleep(interval_minutes * 60)
            continue

        # 3. Calculate momentum and macro trends
        price_mom = price_momentum(price_df)
        macro_mom = macro_momentum(macro_df)

        # 4. Check if signals are aligned
        aligned = is_aligned(price_mom, macro_mom)

        # 5. Generate trade signal
        signal = generate_trade_signal(aligned)

        # 6. Output
        current_price = price_df['Close'].iloc[-1]
        print(f"SPY Price: {current_price:.2f}")
        print(f"Price Momentum: {price_mom['momentum'].iloc[-1]}")
        print(f"Macro Trend: {macro_mom['positive_trend'].iloc[-1]}")
        print(f"Signal: {signal}")

        # Wait until next cycle
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    run_live_loop()
