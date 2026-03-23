def add_features(df):
    # Basis
    df['return'] = df['Close'].pct_change()

    # Technische Features
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['volatility'] = df['return'].rolling(10).std()

    # 🎯 Ziel: Return von morgen
    df['target'] = df['return'].shift(-1)

    return df.dropna()