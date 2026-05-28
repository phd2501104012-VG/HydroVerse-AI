from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import CFG
from utils.logger import log


class AnomalyDetector:
    def __init__(self):
        self._baselines: Dict[str, Dict] = {}
        self._anomaly_scores: Dict[str, pd.Series] = {}

    def compute_baseline(
        self,
        historical: pd.Series,
        window_days: int = 365,
    ) -> Dict:
        recent = historical.tail(window_days)
        return {
            "mean": float(recent.mean()),
            "std": float(recent.std()),
            "p05": float(recent.quantile(0.05)),
            "p25": float(recent.quantile(0.25)),
            "p50": float(recent.quantile(0.50)),
            "p75": float(recent.quantile(0.75)),
            "p95": float(recent.quantile(0.95)),
            "n": len(recent),
        }

    def z_score_anomaly(
        self,
        value: float,
        baseline: Dict,
    ) -> float:
        if baseline["std"] == 0:
            return 0.0
        return (value - baseline["mean"]) / baseline["std"]

    def percentile_anomaly(
        self,
        value: float,
        baseline: Dict,
    ) -> float:
        if value <= baseline["p05"]:
            return -2.0
        elif value <= baseline["p25"]:
            return -1.0
        elif value <= baseline["p75"]:
            return 0.0
        elif value <= baseline["p95"]:
            return 1.0
        return 2.0

    def detect_anomalies(
        self,
        current: pd.DataFrame,
        historical: Dict[str, pd.Series],
        value_col: str,
        district_col: str = "district",
    ) -> pd.DataFrame:
        results = []
        for _, row in current.iterrows():
            district = row[district_col]
            value = row.get(value_col)

            if value is None or pd.isna(value):
                continue

            if district not in historical:
                continue

            key = f"{district}_{value_col}"
            if key not in self._baselines:
                self._baselines[key] = self.compute_baseline(historical[district])

            baseline = self._baselines[key]
            z_score = self.z_score_anomaly(value, baseline)
            p_score = self.percentile_anomaly(value, baseline)

            anomaly_class = "Normal"
            if abs(z_score) >= 3:
                anomaly_class = "Extreme"
            elif abs(z_score) >= 2:
                anomaly_class = "Severe"
            elif abs(z_score) >= 1.5:
                anomaly_class = "Moderate"

            results.append({
                "district": district,
                "variable": value_col,
                "value": value,
                "baseline_mean": baseline["mean"],
                "z_score": round(z_score, 3),
                "percentile_score": p_score,
                "anomaly_class": anomaly_class,
                "timestamp": datetime.utcnow().isoformat(),
            })

        return pd.DataFrame(results)

    def update_anomaly(
        self,
        district: str,
        variable: str,
        value: float,
        historical: pd.Series,
    ) -> Dict:
        key = f"{district}_{variable}"
        if key not in self._baselines:
            self._baselines[key] = self.compute_baseline(historical)

        baseline = self._baselines[key]
        z_score = self.z_score_anomaly(value, baseline)

        anomaly_class = "Normal"
        if abs(z_score) >= 3:
            anomaly_class = "Extreme"
        elif abs(z_score) >= 2:
            anomaly_class = "Severe"
        elif abs(z_score) >= 1.5:
            anomaly_class = "Moderate"

        return {
            "district": district,
            "variable": variable,
            "value": value,
            "z_score": round(z_score, 3),
            "anomaly_class": anomaly_class,
            "baseline_mean": baseline["mean"],
            "baseline_std": baseline["std"],
        }

    def get_hotspot_districts(self, anomaly_df: pd.DataFrame, threshold: float = 2.0) -> List[str]:
        severe = anomaly_df[anomaly_df["z_score"].abs() >= threshold]
        return severe["district"].unique().tolist()
