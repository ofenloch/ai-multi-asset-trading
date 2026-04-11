
# For development, testing and debugging we disable all randomness:
random_seed = 42
import random
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)
import torch
torch.manual_seed(random_seed)
torch.use_deterministic_algorithms(True)

import yfinance as yf

from sklearn.preprocessing import StandardScaler

from config import *
from utils.data_loader import load_data
from utils.features import add_features
from utils.sequences import create_sequences
from models.lstm_model import MultiAssetLSTM

# ------------------------------
# 0. Imports & Config
# ------------------------------
print("Tickers:             ", TICKERS)
print("Start date:          ", START_DATE)
print("End date:            ", END_DATE)
print("Sequence length:     ", SEQ_LEN)
print("Train split:         ", TRAIN_SPLIT)
print("Top K:               ", TOP_K)
print("Transaction cost:    ", TRANSACTION_COST)
print("torch version:       ", torch.__version__)
print("sklearn version:     ", StandardScaler.__module__.split('.')[0])
print("numpy version:       ", np.__version__)
print("yfinance version:    ", yf.__version__)
print("Random seed:         ", random_seed)


# -----------------------------
# 1. Daten laden
# -----------------------------
data = load_data(TICKERS, START_DATE, END_DATE)

all_data = {}

# -----------------------------
# 2. Features + Sequenzen pro Asset
# -----------------------------
for t in TICKERS:
    df = add_features(data[t])

    feature_cols = ['return','ma_5','ma_20','volatility','Volume']
    features = df[feature_cols].values
    target = df['target'].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X, y = create_sequences(features, target, SEQ_LEN)

    all_data[t] = (X, y)

# -----------------------------
# 3. Trainingsdaten kombinieren
# -----------------------------
X_train_all = []
y_train_all = []

for t in TICKERS:
    X, y = all_data[t]
    split = int(len(X) * TRAIN_SPLIT)

    X_train_all.append(X[:split])
    y_train_all.append(y[:split])

X_train = np.concatenate(X_train_all)
y_train = np.concatenate(y_train_all)

# -----------------------------
# 4. Modell
# -----------------------------
model = MultiAssetLSTM(input_size=X_train.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

# -----------------------------
# 5. Training
# -----------------------------
for epoch in range(10):
    optimizer.zero_grad()

    preds = model(X_train_t).squeeze()
    loss = criterion(preds, y_train_t)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: {loss.item()}")

# -----------------------------
# 6. Predictions pro Asset
# -----------------------------
model.eval()

results = {}

for t in TICKERS:
    X, y = all_data[t]
    split = int(len(X) * TRAIN_SPLIT)

    X_test = X[split:]
    y_test = y[split:]

    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    logits = model(X_test_t).detach().numpy().flatten()
    probs = 1 / (1 + np.exp(-logits))

    results[t] = (probs, y_test)

# -----------------------------
# 7. Portfolio Backtest
# -----------------------------
min_len = min(len(results[t][0]) for t in TICKERS)

capital = 1.0
history = []

for i in range(min_len):
    signals = []
    returns = []

    for t in TICKERS:
        probs, rets = results[t]

        signals.append(probs[i])
        returns.append(rets[i])

    signals = np.array(signals)
    returns = np.array(returns)

    # nur starke Signale hndeln
    strong = signals > 0.55

    if strong.sum() == 0:
        history.append(capital)
        continue

    filtered_signals = signals.copy()
    filtered_signals[~strong] = 0

    selected = np.argsort(filtered_signals)[-TOP_K:]

    # Problem 2: Kein Confidence-Filter
    # 👉 Du nutzt:
    #   signals = probs
    # Aber:
    #      0.51 = sehr unsicher
    #      0.60 = viel besser
    # 👉 Du behandelst beides gleich → Fehler

    # Problem 3: Keine Edge-Selektion
    # 👉 Ein gutes Quant-System macht:
    # ❌ nicht:
    # „immer handeln“
    # ✅ sondern:
    # „nur handeln, wenn ich Vorteil habe“

    weights = np.zeros(len(TICKERS))
    weights[selected] = 1 / TOP_K

    portfolio_return = np.sum(weights * returns) - TRANSACTION_COST

    capital *= (1 + portfolio_return)
    history.append(capital)

# -----------------------------
# 8. Ergebnis
# -----------------------------
print("Avg signals:", signals.mean())
print("Max signals:", signals.max())
print("Final capital:", capital)