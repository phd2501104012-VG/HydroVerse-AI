from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config import CFG, constants
from hazards.categories import HazardClassifier
from utils.logger import log


class AlertEngine:
    def __init__(self):
        self.classifier = HazardClassifier()
        self._active_alerts: pd.DataFrame = pd.DataFrame()
        self._alert_history: List[Dict] = []
        self._cooldowns: Dict[str, datetime] = {}

    def evaluate(
        self,
        hazard_data: Dict[str, pd.DataFrame],
        district: str,
        min_severity: float = 25.0,
    ) -> pd.DataFrame:
        rows = []

        for hazard_name, severity_col in [
            ("flood", "flood_severity"),
            ("drought", "drought_severity"),
            ("heatwave", "heatwave_severity"),
            ("agri_stress", "agri_severity"),
            ("compound", "compound_severity"),
        ]:
            if hazard_name not in hazard_data:
                continue

            df = hazard_data[hazard_name]
            if df.empty or severity_col not in df.columns:
                continue

            latest = df.iloc[-1]
            severity = float(latest.get(severity_col, 0))

            if severity < min_severity:
                continue

            cls = self.classifier.classify(severity)
            cooldown_key = f"{district}_{hazard_name}"
            now = datetime.utcnow()

            if cooldown_key in self._cooldowns:
                if now - self._cooldowns[cooldown_key] < timedelta(hours=CFG.realtime.alert_cooldown_hours):
                    continue

            self._cooldowns[cooldown_key] = now

            row = {
                "timestamp": now.isoformat(),
                "district": district,
                "hazard": hazard_name,
                "severity": round(severity, 1),
                "risk_class": cls,
                "recommended_action": constants.ACTION_PLAYBOOK.get(hazard_name, {}).get(cls, "Consult local protocol."),
                "confidence": min(severity / 100 + 0.3, 0.95),
            }
            rows.append(row)
            self._alert_history.append(row)

        alerts_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        self._active_alerts = alerts_df
        return alerts_df

    def evaluate_all_districts(
        self,
        hazard_data_all: Dict[str, Dict[str, pd.DataFrame]],
        districts: List[str],
    ) -> pd.DataFrame:
        all_alerts = []
        for d in districts:
            alerts = self.evaluate(hazard_data_all.get(d, {}), d)
            if not alerts.empty:
                all_alerts.append(alerts)

        if all_alerts:
            self._active_alerts = pd.concat(all_alerts, ignore_index=True).sort_values(
                "severity", ascending=False
            ).reset_index(drop=True)
        else:
            self._active_alerts = pd.DataFrame()

        return self._active_alerts

    @property
    def active_alerts(self) -> pd.DataFrame:
        return self._active_alerts

    @property
    def alert_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._alert_history)

    def clear_cooldowns(self):
        self._cooldowns.clear()

    def get_summary(self) -> Dict:
        if self._active_alerts.empty:
            return {
                "total": 0,
                "by_severity": {},
                "by_hazard": {},
                "by_district": {},
                "critical": 0,
            }

        return {
            "total": len(self._active_alerts),
            "by_severity": self._active_alerts["risk_class"].value_counts().to_dict(),
            "by_hazard": self._active_alerts["hazard"].value_counts().to_dict(),
            "by_district": self._active_alerts["district"].value_counts().to_dict(),
            "critical": len(self._active_alerts[self._active_alerts["severity"] >= 75]),
        }
