import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import *
from utils.data_loader import load_data
from utils.features import add_features
from utils.sequences import create_sequences
from models.lstm_model import MultiAssetLSTM
from backtest.portfolio import backtest

# Daten laden
data = load_data(TICKERS, START_DATE, END_DATE)

all_X, all_y = [], []

for i, t in enumerate(TICKERS):
    df = add_features(data[t])

    feature_cols = ['return','ma_5','ma_20','volatility','Volume']
    features = df[feature_cols].values
    target = df['target'].values

    # Skalierung
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Asset-ID hinzufügen
    asset_id = np.full((len(features), 1), i)
    features = np.hstack([features, asset_id])

    # Sequenzen erstellen
    X, y = create_sequences(features, target, SEQ_LEN)

    all_X.append(X)
    all_y.append(y)

X = np.concatenate(all_X)
y = np.concatenate(all_y)

# Train/Test Split
split = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Modell
model = MultiAssetLSTM(input_size=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 🎯 Klassifikation!
criterion = torch.nn.BCEWithLogitsLoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

# Training
for epoch in range(10):
    optimizer.zero_grad()

    preds = model(X_train_t).squeeze()
    loss = criterion(preds, y_train_t)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: {loss.item()}")

# Prediction
model.eval()
X_test_t = torch.tensor(X_test, dtype=torch.float32)

logits = model(X_test_t).detach().numpy().flatten()

# Wahrscheinlichkeiten (Up/Down)
probs = 1 / (1 + np.exp(-logits))

n_assets = len(TICKERS)

# Länge anpassen
usable_length = (len(probs) // n_assets) * n_assets

probs = probs[:usable_length]
y_test = y_test[:usable_length]

# reshape
probs = probs.reshape(-1, n_assets)
returns = y_test.reshape(-1, n_assets)  # 👉 echte Returns!

# 👉 Trading-Entscheidung aus Wahrscheinlichkeiten
signals = (probs > 0.55).astype(int)

# Backtest mit echten Returns
history = backtest(signals, returns, TOP_K, THRESHOLD, TRANSACTION_COST)

print("Final capital:", history[-1])