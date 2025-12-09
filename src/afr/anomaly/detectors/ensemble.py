"""Ensemble voting detector composed of child detectors."""

from __future__ import annotations

from typing import Iterable, List, Mapping

import numpy as np

from ..base import BaseDetector
from ..engine import AnomalyEngine


class EnsembleDetector(BaseDetector):
    """Wraps multiple detectors and votes on anomalies."""

    def __init__(
        self,
        children: Iterable[BaseDetector],
        vote: str = "any",
        name: str = "ensemble",
    ) -> None:
        super().__init__(name=name, threshold=None)
        self.children = list(children)
        self.vote = vote
        self.metadata.update({"vote": vote, "children": [c.name for c in self.children]})

    def score(self, X) -> float:
        arr = np.asarray(X, dtype=float)
        if arr.size == 0 or not self.children:
            return 0.0
        scores = [float(det.score(arr)) for det in self.children]
        return max(scores) if scores else 0.0

    def predict(self, X, *, score: float | None = None) -> bool:
        arr = np.asarray(X, dtype=float)
        votes: List[bool] = []
        for det in self.children:
            s = float(det.score(arr))
            votes.append(det.predict(arr, score=s))
        if not votes:
            return False
        if self.vote == "majority":
            return sum(votes) >= (len(votes) / 2.0)
        if self.vote == "all":
            return all(votes)
        return any(votes)


def build_ensemble_from_config(cfg: Mapping[str, object]) -> EnsembleDetector:
    children_cfg = cfg.get("detectors", []) or cfg.get("children", [])
    children: list[BaseDetector] = []
    for child in children_cfg:
        det_type = str(child.get("type", "")).lower()  # type: ignore[union-attr]
        factory = AnomalyEngine._registry.get(det_type)  # pylint: disable=protected-access
        if not factory:
            raise ValueError(f"Unknown detector type '{det_type}' for ensemble child")
        children.append(factory(child))
    vote = str(cfg.get("vote", "any"))
    name = cfg.get("name", "ensemble")
    return EnsembleDetector(children=children, vote=vote, name=name)
