import numpy as np
import pandas as pd
from typing import List, Union, Optional

from config import CFG, constants


class HazardClassifier:
    def __init__(self):
        self.classes = constants.RISK_CLASSES
        self.thresholds = CFG.hazard.risk_thresholds

    def classify(self, severity: Union[float, pd.Series, np.ndarray]) -> Union[str, pd.Series]:
        if isinstance(severity, (int, float)):
            return self._classify_scalar(severity)
        return pd.cut(
            severity,
            bins=[-1] + self.thresholds[1:] + [101],
            labels=self.classes,
            include_lowest=True,
        ).astype(str)

    def _classify_scalar(self, severity: float) -> str:
        for i, thr in enumerate(self.thresholds):
            if severity <= thr:
                return self.classes[i]
        return self.classes[-1]

    def get_severity_color(self, severity: float) -> str:
        cls = self.classify(severity)
        return constants.RISK_COLORS.get(cls, "#22c55e")

    def get_class_color(self, cls: str) -> str:
        return constants.RISK_COLORS.get(cls, "#22c55e")

    def get_action(self, hazard: str, severity: float) -> str:
        cls = self.classify(severity)
        playbook = constants.ACTION_PLAYBOOK.get(hazard, {})
        return playbook.get(cls, "Consult local protocol.")
