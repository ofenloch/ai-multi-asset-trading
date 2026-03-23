import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import *
from utils.data_loader import load_data
from utils.features import add_features
from utils.sequences import create_sequences
from models.lstm_model import MultiAssetLSTM
from backtest.portfolio import backtest

data = load_data(TICKERS, START_DATE, END_DATE)

all_X, all_y = [], []

for i, t in enumerate(TICKERS):
    df = add_features(data[t])
    features = df[['return','ma_5','ma_20','volatility','Volume']].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    asset_id = np.full((len(features), 1), i)
    features = np.hstack([features, asset_id])

    X, y = create_sequences(features, SEQ_LEN)

    all_X.append(X)
    all_y.append(y)

X = np.concatenate(all_X)
y = np.concatenate(all_y)

split = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = MultiAssetLSTM(input_size=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

for epoch in range(10):
    optimizer.zero_grad()
    preds = model(X_train_t).squeeze()
    loss = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: {loss.item()}")

model.eval()
X_test_t = torch.tensor(X_test, dtype=torch.float32)
preds = model(X_test_t).detach().numpy()

history = backtest(preds, y_test, TOP_K, THRESHOLD, TRANSACTION_COST)
print("Final capital:", history[-1])
