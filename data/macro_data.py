from pandas_datareader import data as web
import pandas as pd

def fetch_macro_indicator(indicator, start, end):
    df = web.DataReader(indicator, 'fred', start, end)
    # Ensure datetime index with no timezone info
    df.index = pd.to_datetime(df.index).normalize()
    return df
