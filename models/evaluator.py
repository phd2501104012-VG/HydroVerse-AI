from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

from utils.logger import log


class ModelEvaluator:
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "nse": float(self._nash_sutcliffe(y_true, y_pred)),
            "mape": float(self._mape(y_true, y_pred)),
            "bias": float(np.mean(y_pred - y_true)),
            "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
        }

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        unique = len(set(y_true))

        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)) if unique > 1 else 0.5,
            "accuracy": float(np.mean(y_pred == y_true)),
            "true_positives": int((y_pred == 1) & (y_true == 1)).sum(),
            "false_positives": int((y_pred == 1) & (y_true == 0)).sum(),
            "false_negatives": int((y_pred == 0) & (y_true == 1)).sum(),
            "true_negatives": int((y_pred == 0) & (y_true == 0)).sum(),
        }

    def _nash_sutcliffe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10))

    def _mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def best_model(self, results: Dict[str, Dict], metric: str = "rmse", lower_is_better: bool = True) -> str:
        if lower_is_better:
            return min(results, key=lambda k: results[k].get(metric, float("inf")))
        return max(results, key=lambda k: results[k].get(metric, float("-inf")))

    def model_ranking(self, results: Dict[str, Dict]) -> pd.DataFrame:
        rows = []
        for name, metrics in results.items():
            row = {"model": name}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("rmse")
