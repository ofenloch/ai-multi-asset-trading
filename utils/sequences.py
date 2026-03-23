import numpy as np

def create_sequences(features, target, seq_len):
    X, y = [], []

    for i in range(len(features) - seq_len - 1):
        X.append(features[i:i+seq_len])

        future_return = target[i+seq_len]

        # Klassifikation: 1 = steigt, 0 = fällt
        # y.append(1 if future_return > 0 else 0)
        # Targets als echte Returns speichern
        y.append(future_return)
    return np.array(X), np.array(y)