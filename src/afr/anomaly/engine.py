"""Unified anomaly engine with pluggable detectors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from .base import BaseDetector
from ..storage.base import StorageBackend
from ..registry.plugins.base import RegistryPlugin, NullRegistryPlugin

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    detector: str
    score: float
    is_anomaly: bool
    threshold: float | None = None
    started_at: datetime | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "score": self.score,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "extras": self.extras,
        }


class AnomalyEngine:
    """Coordinates detectors and routes events to storage/plugins."""

    _registry: MutableMapping[str, Callable[[Mapping[str, Any]], BaseDetector]] = {}

    def __init__(
        self,
        detectors: Iterable[BaseDetector],
        storage: StorageBackend | None = None,
        registry_plugin: RegistryPlugin | None = None,
    ) -> None:
        self.detectors = list(detectors)
        self.storage = storage
        self.registry_plugin = registry_plugin or NullRegistryPlugin()

    @classmethod
    def register_detector(cls, name: str, factory: Callable[[Mapping[str, Any]], BaseDetector]) -> None:
        cls._registry[name.lower()] = factory

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        storage: StorageBackend | None = None,
        registry_plugin: RegistryPlugin | None = None,
    ) -> "AnomalyEngine":
        detectors: list[BaseDetector] = []
        for detector_cfg in config.get("detectors", []):
            det_type = str(detector_cfg.get("type") or detector_cfg.get("name", "")).lower()
            if not det_type:
                raise ValueError("Detector config requires a 'type' field.")
            factory = cls._registry.get(det_type)
            if not factory:
                raise ValueError(f"Unknown detector type '{det_type}'. Registered detectors: {sorted(cls._registry)}")
            detectors.append(factory(detector_cfg))
        return cls(detectors=detectors, storage=storage, registry_plugin=registry_plugin)

    def fit(self, X: Sequence[Sequence[float]] | np.ndarray) -> None:
        """Train any detectors that require historical context."""

        arr = np.asarray(X, dtype=float)
        for det in self.detectors:
            try:
                det.fit(arr)
            except Exception:  # pragma: no cover - defensive logging path
                logger.exception("Detector %s failed to fit", det.name)

    def evaluate_window(
        self,
        window: Sequence[float] | np.ndarray,
        sample_rate: float,
        *,
        started_at: datetime | None = None,
    ) -> List[DetectionResult]:
        """Evaluate a single window and persist any anomalies."""

        arr = np.asarray(window, dtype=float)
        started_at = started_at or datetime.now(timezone.utc)
        results: list[DetectionResult] = []

        if self.storage:
            self.storage.store_window(arr, sample_rate=sample_rate, started_at=started_at)

        for det in self.detectors:
            if hasattr(det, "update_context"):
                try:
                    det.update_context(sample_rate=sample_rate)
                except Exception:  # pragma: no cover - defensive logging path
                    logger.exception("Detector %s failed to update context", det.name)
            elif hasattr(det, "sample_rate"):
                setattr(det, "sample_rate", sample_rate)

            score = float(det.score(arr))
            is_anomaly = det.predict(arr, score=score)
            result = DetectionResult(
                detector=det.name,
                score=score,
                is_anomaly=is_anomaly,
                threshold=det.threshold,
                started_at=started_at,
                extras=det.describe(),
            )
            results.append(result)
            if is_anomaly and self.storage:
                self.storage.store_event(
                    detector=det.name,
                    score=score,
                    threshold=det.threshold,
                    started_at=started_at,
                    metadata={"extras": det.describe()},
                )
            if is_anomaly:
                self.registry_plugin.on_detection(result, window=arr)

        return results

    def evaluate_batch(
        self,
        windows: Iterable[Sequence[float] | np.ndarray],
        sample_rate: float,
    ) -> List[DetectionResult]:
        """Evaluate multiple windows and return flattened detection results."""

        results: list[DetectionResult] = []
        for window in windows:
            results.extend(self.evaluate_window(window, sample_rate=sample_rate))
        return results


# Import detectors and register factories
from .detectors.statistical_zscore import StatisticalZScoreDetector  # noqa: E402
from .detectors.spectral_bandpower import SpectralBandpowerDetector  # noqa: E402
from .detectors.change_point import ChangePointDetector  # noqa: E402
from .detectors.autoencoder import AutoencoderDetector  # noqa: E402
from ..detection.statistical import MADDetector, RollingMeanVarianceDetector  # noqa: E402
from ..detection.machine_learning import IsolationForestDetector, OneClassSVMDetector  # noqa: E402


def _safe(factory: Callable[[Mapping[str, Any]], BaseDetector]) -> Callable[[Mapping[str, Any]], BaseDetector]:
    def wrapper(cfg: Mapping[str, Any]) -> BaseDetector:
        try:
            return factory(cfg)
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ValueError(str(exc)) from exc

    return wrapper


def _register_defaults() -> None:
    AnomalyEngine.register_detector(
        "zscore",
        lambda cfg: StatisticalZScoreDetector(
            threshold=cfg.get("threshold", 3.5),
            window=cfg.get("window"),
            name=cfg.get("name", "zscore"),
        ),
    )
    AnomalyEngine.register_detector(
        "statisticalzscoredetector",
        lambda cfg: StatisticalZScoreDetector(
            threshold=cfg.get("threshold", 3.5),
            window=cfg.get("window"),
            name=cfg.get("name", "zscore"),
        ),
    )
    AnomalyEngine.register_detector(
        "spectralbandpowerdetector",
        lambda cfg: SpectralBandpowerDetector(
            bands=cfg.get("bands"),
            threshold=cfg.get("threshold"),
            name=cfg.get("name", "bandpower"),
        ),
    )
    AnomalyEngine.register_detector(
        "bandpower",
        lambda cfg: SpectralBandpowerDetector(
            bands=cfg.get("bands"),
            threshold=cfg.get("threshold"),
            name=cfg.get("name", "bandpower"),
        ),
    )
    AnomalyEngine.register_detector(
        "changepointdetector",
        lambda cfg: ChangePointDetector(
            threshold=cfg.get("threshold", 5.0),
            drift=cfg.get("drift", 0.0),
            name=cfg.get("name", "changepoint"),
        ),
    )
    AnomalyEngine.register_detector(
        "cusum",
        lambda cfg: ChangePointDetector(
            threshold=cfg.get("threshold", 5.0),
            drift=cfg.get("drift", 0.0),
            name=cfg.get("name", "changepoint"),
        ),
    )
    AnomalyEngine.register_detector(
        "autoencoder",
        lambda cfg: AutoencoderDetector(
            threshold=cfg.get("threshold", 0.05),
            encoding_dim=cfg.get("encoding_dim", 8),
            epochs=cfg.get("epochs", 10),
            name=cfg.get("name", "autoencoder"),
        ),
    )
    AnomalyEngine.register_detector(
        "mad",
        lambda cfg: MADDetector(
            threshold=cfg.get("threshold", 3.5),
            name=cfg.get("name", "mad"),
        ),
    )
    AnomalyEngine.register_detector(
        "rolling_mean_variance",
        lambda cfg: RollingMeanVarianceDetector(
            threshold=cfg.get("threshold", 3.0),
            window=cfg.get("window", 128),
            name=cfg.get("name", "rolling_mean_variance"),
        ),
    )
    AnomalyEngine.register_detector(
        "isolation_forest",
        _safe(
            lambda cfg: IsolationForestDetector(
                contamination=cfg.get("contamination", 0.05),
                n_estimators=cfg.get("n_estimators", 100),
                threshold=cfg.get("threshold", 0.0),
                name=cfg.get("name", "isolation_forest"),
            )
        ),
    )
    AnomalyEngine.register_detector(
        "one_class_svm",
        _safe(
            lambda cfg: OneClassSVMDetector(
                kernel=cfg.get("kernel", "rbf"),
                nu=cfg.get("nu", 0.05),
                gamma=cfg.get("gamma", "scale"),
                threshold=cfg.get("threshold", 0.0),
                name=cfg.get("name", "one_class_svm"),
            )
        ),
    )


_register_defaults()
