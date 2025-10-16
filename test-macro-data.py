from pandas_datareader import data as web
import datetime

start = datetime.datetime.today() - datetime.timedelta(days=60)
end = datetime.datetime.today()

# Try these symbols one by one to see if any return data:
symbols = ['UNRATE', 'CPIAUCSL', 'FEDFUNDS']

for sym in symbols:
    df = web.DataReader(sym, 'fred', start, end)
    print(f"\nSymbol: {sym}")
    print(df.head())
