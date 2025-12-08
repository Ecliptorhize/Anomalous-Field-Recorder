"""Autoencoder-based anomaly detector."""

from __future__ import annotations

import numpy as np

from ..base import BaseDetector

try:  # pragma: no cover - dependency is optional
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - fallback when torch unavailable
    torch = None
    nn = None


if nn is not None:
    class _TinyAutoencoder(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, encoding_dim: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(encoding_dim, input_dim))

        def forward(self, x):  # type: ignore[override]
            latent = self.encoder(x)
            return self.decoder(latent)
else:  # pragma: no cover - torch not available
    class _TinyAutoencoder:  # type: ignore[misc]
        def __init__(self, input_dim: int, encoding_dim: int) -> None:
            raise ImportError("torch is required for AutoencoderDetector")


class AutoencoderDetector(BaseDetector):
    """Uses reconstruction error from a small autoencoder."""

    def __init__(
        self,
        threshold: float = 0.05,
        encoding_dim: int = 8,
        epochs: int = 10,
        name: str = "autoencoder",
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.model: _TinyAutoencoder | None = None
        self.trained = False
        self.input_dim: int | None = None
        self.metadata.update({"encoding_dim": encoding_dim, "epochs": epochs})

    def _ensure_model(self, input_dim: int) -> None:
        if torch is None:
            return
        if self.model is None or self.input_dim != input_dim:
            self.model = _TinyAutoencoder(input_dim, self.encoding_dim)
            self.input_dim = input_dim

    def fit(self, X) -> None:
        if torch is None:
            return
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return
        arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
        self._ensure_model(arr.shape[1])
        if not self.model:
            return
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        data = torch.tensor(arr, dtype=torch.float32)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            recon = self.model(data)
            loss = loss_fn(recon, data)
            loss.backward()
            optimizer.step()
        self.trained = True

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0:
            return 0.0
        vector = arr.reshape(1, -1)

        if torch is None:
            # Fallback: use variance as a proxy reconstruction error
            return float(np.var(vector))

        self._ensure_model(vector.shape[1])
        if not self.model:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            batch = torch.tensor(vector, dtype=torch.float32)
            recon = self.model(batch)
            loss = torch.mean((batch - recon) ** 2).item()
        return float(loss)
