import yfinance as yf

def fetch_spy_price_data(period='60d', interval='1h'):
    spy = yf.Ticker('SPY')
    return spy.history(period=period, interval=interval)