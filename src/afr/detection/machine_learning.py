"""Machine learning-based anomaly detectors with sensible fallbacks."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from afr.anomaly.base import BaseDetector

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
except ModuleNotFoundError:  # pragma: no cover - fallback when scikit-learn missing
    IsolationForest = None  # type: ignore[assignment]
    OneClassSVM = None  # type: ignore[assignment]

try:  # pragma: no cover - torch is optional
    from afr.anomaly.detectors.autoencoder import AutoencoderDetector as _AutoencoderDetector
except Exception:  # pragma: no cover - defensive import
    _AutoencoderDetector = None  # type: ignore[assignment]


class IsolationForestDetector(BaseDetector):
    """IsolationForest-based detector using negative score as anomaly magnitude."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        threshold: float = 0.0,
        name: str = "isolation_forest",
        random_state: int | None = 42,
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        if IsolationForest is None:
            raise ImportError("scikit-learn is required for IsolationForestDetector")
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.trained = False
        self.metadata.update({"contamination": contamination, "n_estimators": n_estimators})

    def _reshape(self, X: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, X: Sequence[float] | np.ndarray) -> None:
        data = self._reshape(X)
        if data.size == 0:
            return
        self.model.fit(data)
        self.trained = True

    def score(self, X: Sequence[float] | np.ndarray) -> float:
        data = self._reshape(X)
        if data.size == 0:
            return 0.0
        if not self.trained:
            self.fit(data)
        scores = -self.model.score_samples(data)  # higher means more anomalous
        return float(np.max(scores))


class OneClassSVMDetector(BaseDetector):
    """One-Class SVM detector leveraging decision_function outputs."""

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str | float = "scale",
        threshold: float = 0.0,
        name: str = "one_class_svm",
    ) -> None:
        super().__init__(name=name, threshold=threshold)
        if OneClassSVM is None:
            raise ImportError("scikit-learn is required for OneClassSVMDetector")
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.trained = False
        self.metadata.update({"kernel": kernel, "nu": nu, "gamma": gamma})

    def _reshape(self, X: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def fit(self, X: Sequence[float] | np.ndarray) -> None:
        data = self._reshape(X)
        if data.size == 0:
            return
        self.model.fit(data)
        self.trained = True

    def score(self, X: Sequence[float] | np.ndarray) -> float:
        data = self._reshape(X)
        if data.size == 0:
            return 0.0
        if not self.trained:
            try:
                self.fit(data)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("OneClassSVM failed to fit; returning neutral score")
                return 0.0
        scores = -self.model.decision_function(data)
        return float(np.max(scores))


class AutoencoderDetector(BaseDetector):
    """PyTorch autoencoder detector exposed under the detection namespace."""

    def __init__(
        self,
        threshold: float = 0.05,
        encoding_dim: int = 8,
        epochs: int = 10,
        name: str = "autoencoder",
    ) -> None:
        if _AutoencoderDetector is None:
            raise ImportError("torch is required for AutoencoderDetector")
        # Delegate to the existing implementation to avoid duplication
        self._delegate = _AutoencoderDetector(
            threshold=threshold,
            encoding_dim=encoding_dim,
            epochs=epochs,
            name=name,
        )
        super().__init__(name=name, threshold=threshold, metadata=self._delegate.metadata)  # type: ignore[arg-type]

    def fit(self, X: Sequence[float] | np.ndarray) -> None:
        self._delegate.fit(X)

    def score(self, X: Sequence[float] | np.ndarray) -> float:
        return float(self._delegate.score(X))

    def predict(self, X: Sequence[float] | np.ndarray, *, score: float | None = None) -> bool:
        return bool(self._delegate.predict(X, score=score))

    def describe(self) -> dict[str, Any]:
        return self._delegate.describe()
