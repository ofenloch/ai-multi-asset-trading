import yfinance as yf

def load_data(tickers, start, end):
    data = {}
    for t in tickers:
        print(f"Loading data for {t} from {start} to {end} ...")
        df = yf.download(t, start=start, end=end)
        data[t] = df
    return data
