import numpy as np

def backtest(preds, targets, top_k, threshold, cost):
    capital = 1.0
    history = []

    for t in range(len(preds)):
        probs = preds[t]

        # nur starke Signale
        mask = probs > 0.55

        if mask.sum() == 0:
            history.append(capital)
            continue

        # Top-K auswählen
        selected = np.argsort(probs)[-top_k:]

        weights = np.zeros_like(probs)
        weights[selected] = 1 / top_k

        # Targets: 1 = Gewinn, 0 = Verlust
        returns = np.where(targets[t] == 1, 0.01, -0.01)

        portfolio_return = np.sum(weights * returns)
        portfolio_return -= cost

        capital *= (1 + portfolio_return)
        history.append(capital)

    return history