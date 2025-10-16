import os
import zipfile
from pathlib import Path

# Define the folder structure and file content
project_structure = {
    "spy-momentum-strategy/": {
        "README.md": "# SPY Momentum Strategy\n\nTrading strategy using SPY options with multi-layer momentum confirmation.",
        "requirements.txt": "yfinance\npandas\npandas_datareader\nnumpy\nPyYAML\nmatplotlib",
        "main.py": "# Entry point for the live strategy\n\nif __name__ == '__main__':\n    print('Running strategy...')",
        "backtest.py": "# Backtesting engine\n\ndef run_backtest():\n    print('Running backtest...')\n\nif __name__ == '__main__':\n    run_backtest()",
        "config/settings.yaml": "data_source: 'yahoo'\nmacro_indicators:\n  - 'LEI'\nrisk:\n  stop_loss_pct: 0.05\ncost_control:\n  max_transaction_cost: 0.01\nmomentum:\n  short_window: 20\n  long_window: 50",
        "data/yahoo_data.py": "import yfinance as yf\n\ndef fetch_spy_price_data(period='60d', interval='1h'):\n    spy = yf.Ticker('SPY')\n    return spy.history(period=period, interval=interval)",
        "data/macro_data.py": "from pandas_datareader import data as web\n\ndef fetch_macro_indicator(indicator, start, end):\n    return web.DataReader(indicator, 'fred', start, end)",
        "data/data_manager.py": "# Interface to fetch price and macro data",
        "indicators/momentum.py": "def price_momentum(df, short=20, long=50):\n    df['short_ma'] = df['Close'].rolling(short).mean()\n    df['long_ma'] = df['Close'].rolling(long).mean()\n    df['momentum'] = df['short_ma'] > df['long_ma']\n    return df",
        "indicators/macro_momentum.py": "def macro_momentum(df):\n    df['delta'] = df.diff()\n    df['positive_trend'] = df['delta'] > 0\n    return df",
        "indicators/alignment_checker.py": "def is_aligned(price_mom_df, macro_mom_df):\n    return price_mom_df['momentum'].iloc[-1] and macro_mom_df['positive_trend'].iloc[-1]",
        "strategy/trade_signal.py": "def generate_trade_signal(is_aligned):\n    return 'BUY_CALL_SPREAD' if is_aligned else None",
        "strategy/risk_manager.py": "def apply_risk_controls(current_price, entry_price, stop_loss_pct=0.05):\n    if current_price < entry_price * (1 - stop_loss_pct):\n        return 'EXIT'\n    return None",
        "execution/order_manager.py": "# Placeholder for placing orders",
        "execution/cost_optimizer.py": "def choose_optimal_strike(options_df, budget, target_delta=0.5):\n    options_df['delta_diff'] = (options_df['delta'] - target_delta).abs()\n    sorted_df = options_df.sort_values('delta_diff')\n    for _, row in sorted_df.iterrows():\n        if row['ask'] <= budget:\n            return row\n    return None",
        "utils/logger.py": "import logging\n\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)"
    }
}

# Create zip file
zip_filename = "spy-momentum-strategy.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for path, content in project_structure.items():
        if isinstance(content, dict):
            for file_path, file_content in content.items():
                full_path = os.path.join(path, file_path)
                zipf.writestr(full_path, file_content)
        else:
            zipf.writestr(path, content)

print(f"✅ Created: {zip_filename}")
