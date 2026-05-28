from typing import Optional, Dict, List, Any, Callable
import numpy as np
import pandas as pd

from utils.logger import log


class ModelEnsemble:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}
        self._metrics: Dict[str, Dict] = {}

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        self._models[name] = model
        self._weights[name] = weight

    def set_weights(self, weights: Dict[str, float]):
        total = sum(weights.values())
        self._weights = {k: v / total for k, v in weights.items()}

    def equal_weights(self):
        n = len(self._models)
        self._weights = {name: 1.0 / n for name in self._models}

    def performance_weights(self, metric_name: str = "rmse", lower_is_better: bool = True):
        if not self._metrics:
            log.warning("No metrics available for performance weighting")
            self.equal_weights()
            return

        if lower_is_better:
            scores = {name: 1.0 / (m.get(metric_name, 1e-6) + 1e-6) for name, m in self._metrics.items()}
        else:
            scores = {name: m.get(metric_name, 0) for name, m in self._metrics.items()}

        total = sum(scores.values())
        self._weights = {name: s / total for name, s in scores.items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            return np.zeros(X.shape[0])

        predictions = []
        for name, model in self._models.items():
            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                weight = self._weights.get(name, 1.0 / len(self._models))
                predictions.append(pred * weight)
            except Exception as e:
                log.warning(f"{name} predict failed: {e}")

        if not predictions:
            return np.zeros(X.shape[0])
        return np.sum(predictions, axis=0)

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        predictions = []
        for name, model in self._models.items():
            try:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                pass

        if not predictions:
            return {"mean": np.zeros(X.shape[0]), "std": np.zeros(X.shape[0])}

        preds = np.array(predictions)
        return {
            "mean": np.mean(preds, axis=0),
            "std": np.std(preds, axis=0),
            "p10": np.percentile(preds, 10, axis=0),
            "p90": np.percentile(preds, 90, axis=0),
            "min": np.min(preds, axis=0),
            "max": np.max(preds, axis=0),
        }
