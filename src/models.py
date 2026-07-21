from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        seed: int = 42,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestModel:
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPModel:
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        dropout: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self._model: _MLP | None = None
        self._device: torch.device | None = None
        self.train_losses: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPModel:
        torch.manual_seed(self.seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = _MLP(X.shape[1], self.hidden_dims, self.dropout).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self._device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        self.train_losses = []
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            self.train_losses.append(epoch_loss / len(X))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict()"
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self._device)
            return self._model(X_t).cpu().numpy()
