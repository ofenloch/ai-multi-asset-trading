import numpy as np

def allocate(signals, top_k):
    weights = np.zeros(len(signals))
    top_idx = np.argsort(signals)[-top_k:]
    weights[top_idx] = 1 / top_k
    return weights

def backtest(preds, returns, top_k, threshold, cost):
    capital = 1.0
    history = []

    for i in range(len(preds)):
        signals = preds[i]
        signals = np.where(signals > threshold, signals, 0)

        weights = allocate(signals, top_k)
        portfolio_return = np.sum(weights * returns[i]) - cost

        capital *= (1 + portfolio_return)
        history.append(capital)

    return history
