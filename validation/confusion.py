from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from utils.logger import log


class ConfusionMatrixBuilder:
    def __init__(self):
        pass

    def build(self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None) -> Dict:
        labels = labels or ["No Event", "Event"]
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            "matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-10)),
            "precision": float(tp / (tp + fp + 1e-10)),
            "recall": float(tp / (tp + fn + 1e-10)),
            "f1": float(2 * tp / (2 * tp + fp + fn + 1e-10)),
            "specificity": float(tn / (tn + fp + 1e-10)),
            "false_positive_rate": float(fp / (fp + tn + 1e-10)),
            "false_negative_rate": float(fn / (fn + tp + 1e-10)),
            "positive_predictive_value": float(tp / (tp + fp + 1e-10)),
            "negative_predictive_value": float(tn / (tn + fn + 1e-10)),
            "true_positive_rate": float(tp / (tp + fn + 1e-10)),
            "true_negative_rate": float(tn / (tn + fp + 1e-10)),
            "fowlkes_mallows_index": float(np.sqrt(
                (tp / (tp + fp + 1e-10)) * (tp / (tp + fn + 1e-10))
            )),
            "matthews_correlation": float((
                tp * tn - fp * fn
            ) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10)),
        }

    def multi_class(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

        return {
            "matrix": cm,
            "report": report,
            "per_class": {
                label: report.get(label, {}) for label in labels
            },
        }
