"""Advanced real-time anomaly detection package for AFR."""

from .anomaly.engine import AnomalyEngine, DetectionResult
from .anomaly.base import BaseDetector
from .streaming.service import StreamingService
from .streaming.buffering import WindowBuffer
from .streaming.filters import RealTimeFilterChain
from .streaming.processor import RealTimeProcessor
from .storage.base import StorageBackend
from .storage.sqlite import SQLiteBackend
from .pipeline import PipelineConfig, run_pipeline
from .sensors import SimulatedSensor
from .export import export_run

__all__ = [
    "AnomalyEngine",
    "BaseDetector",
    "DetectionResult",
    "StreamingService",
    "WindowBuffer",
    "RealTimeFilterChain",
    "RealTimeProcessor",
    "StorageBackend",
    "SQLiteBackend",
    "PipelineConfig",
    "run_pipeline",
    "SimulatedSensor",
    "export_run",
]
