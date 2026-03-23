import yfinance as yf

def load_data(tickers, start, end):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end)
        data[t] = df
    return data
