import yfinance as yf

def load_data(tickers, start, end):
    data = {}
    for t in tickers:
        print(f"Loading data for {t} from {start} to {end} ...")
        #df = yf.download(t, start=start, end=end, auto_adjust=False)
        df = yf.Ticker(t).history(interval='1d', start=start, end=end, auto_adjust=False)
        df.to_csv(f"data/{t}.csv")  # Speichern als CSV
        data[t] = df
    return data
