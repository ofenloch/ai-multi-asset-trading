def add_features(df):
    df['return'] = df['Close'].pct_change()
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['volatility'] = df['return'].rolling(10).std()
    return df.dropna()
