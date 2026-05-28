import numpy as np
import pandas as pd

from config import CFG
from hazards.categories import HazardClassifier


class AgriStressDetector:
    def __init__(self):
        self.classifier = HazardClassifier()

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        severity = pd.Series(0.0, index=df.index)
        risk_factors = {}

        # 1. VHI / VCI — Vegetation Health (0-50)
        vhi_score = pd.Series(0.0, index=df.index)
        if "vhi" in df.columns:
            vhi_score = ((50 - df["vhi"].clip(0, 50)) / 50 * 50).clip(0, 50).fillna(0)
        elif "vci" in df.columns:
            vhi_score = ((50 - df["vci"].clip(0, 50)) / 50 * 50).clip(0, 50).fillna(0)
        severity += vhi_score
        risk_factors["vhi"] = vhi_score

        # 2. NDVI anomaly (0-15)
        ndvi_score = pd.Series(0.0, index=df.index)
        if "ndvi_anom" in df.columns:
            ndvi_score = ((-df["ndvi_anom"]).clip(lower=0, upper=2) / 2 * 15).fillna(0)
        elif "ndvi" in df.columns:
            ndvi_score = ((0.3 - df["ndvi"].clip(0, 0.3)) / 0.3 * 15).fillna(0)
        severity += ndvi_score
        risk_factors["ndvi"] = ndvi_score

        # 3. Soil moisture deficit (0-15)
        sm_score = pd.Series(0.0, index=df.index)
        if "soil_moisture_anom" in df.columns:
            sm_score = ((-df["soil_moisture_anom"]).clip(lower=0, upper=2) / 2 * 15).fillna(0)
        elif "soil_moisture" in df.columns:
            sm_base = df["soil_moisture"].fillna(0)
            sm_score = ((0.3 - sm_base.clip(0, 0.3)) / 0.3 * 15).fillna(0)
        severity += sm_score
        risk_factors["soil_moisture"] = sm_score

        # 4. Heat stress from tmax (0-20)
        ht_score = pd.Series(0.0, index=df.index)
        if "tmax" in df.columns:
            ht_score = ((df["tmax"].clip(35, 50) - 35) / 10 * 20).clip(0, 20).fillna(0)
        severity += ht_score
        risk_factors["heat_stress"] = ht_score

        # 5. CDD — Consecutive Dry Days (0-10)
        cdd_score = pd.Series(0.0, index=df.index)
        if "cdd" in df.columns:
            cdd_score = (df["cdd"].clip(0, 30) / 30 * 10).clip(0, 10)
        elif "precip" in df.columns:
            from utils.temporal_utils import compute_cdd_cwd
            runs = compute_cdd_cwd(df["precip"].fillna(0))
            cdd_score = (runs["cdd"].clip(0, 30) / 30 * 10).clip(0, 10)
        severity += cdd_score
        risk_factors["cdd"] = cdd_score

        # 6. Flood occurrence penalty (0-10)
        flood_score = pd.Series(0.0, index=df.index)
        if "flood_severity" in df.columns:
            flood_score = (df["flood_severity"].fillna(0) / 100 * 10)
        severity += flood_score
        risk_factors["flood"] = flood_score

        # 7. Precipitation anomaly (0-10)
        panom_score = pd.Series(0.0, index=df.index)
        if "precip_anom" in df.columns:
            panom_score = ((-df["precip_anom"]).clip(lower=0, upper=2) / 2 * 10).fillna(0)
        severity += panom_score
        risk_factors["precip_anom"] = panom_score

        # Composite agricultural risk classification
        risk_level = pd.Series("Low", index=df.index)
        active_factors = pd.DataFrame(risk_factors)
        n_active = (active_factors > 0).sum(axis=1)
        risk_level = np.where(severity >= 75, "Severe",
                      np.where(severity >= 50, "High",
                      np.where(severity >= 25, "Moderate",
                      np.where(n_active >= 2, "Watch", "Low"))))

        out["agri_severity"] = severity.clip(0, 100).round(1)
        out["agri_event"] = (severity >= 50).astype(int)
        out["agri_risk"] = risk_level
        out["agri_active_factors"] = n_active
        out["agri_class"] = self.classifier.classify(out["agri_severity"])
        return out
