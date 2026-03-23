import numpy as np

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len - 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][0])
    return np.array(X), np.array(y)
