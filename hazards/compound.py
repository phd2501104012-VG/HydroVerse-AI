import numpy as np
import pandas as pd
from typing import Dict, Optional, List

from config import CFG, HazardType
from hazards.categories import HazardClassifier


class CompoundHazardEngine:
    def __init__(self):
        self.classifier = HazardClassifier()
        self.interaction_matrix = {
            ("heatwave", "drought"): {
                "name": "Agricultural Collapse Risk",
                "weight": 1.3,
                "description": "Heatwave + Soil Moisture Deficit → Crop Failure",
            },
            ("extreme_precip", "flood"): {
                "name": "Flash Flood Risk",
                "weight": 1.4,
                "description": "Extreme Rainfall + Saturated Soil → Flash Flooding",
            },
            ("heatwave", "agri_stress"): {
                "name": "Crop Failure Risk",
                "weight": 1.2,
                "description": "High Temperature + Dry Spell → Agricultural Loss",
            },
            ("drought", "heatwave"): {
                "name": "Compound Dry-Hot Extreme",
                "weight": 1.5,
                "description": "Simultaneous drought and heatwave → Multi-sector impact",
            },
            ("flood", "agri_stress"): {
                "name": "Post-Flood Agricultural Stress",
                "weight": 1.1,
                "description": "Flooding followed by soil degradation → Yield loss",
            },
        }

    def compute(
        self,
        df: pd.DataFrame,
        flood_result: pd.DataFrame,
        drought_result: pd.DataFrame,
        heatwave_result: pd.DataFrame,
        agri_result: pd.DataFrame,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        hazards = {
            "flood": flood_result.get("flood_severity", pd.Series(0, index=df.index)),
            "drought": drought_result.get("drought_severity", pd.Series(0, index=df.index)),
            "heatwave": heatwave_result.get("heatwave_severity", pd.Series(0, index=df.index)),
            "agri_stress": agri_result.get("agri_severity", pd.Series(0, index=df.index)),
        }

        extreme_precip = pd.Series(0, index=df.index)
        if "precip" in df.columns:
            p = df["precip"].fillna(0)
            extreme_precip = np.where(p >= CFG.hazard.imd_precip_very_heavy, 
                                      (p / 300 * 100).clip(0, 100), 0)
        hazards["extreme_precip"] = pd.Series(extreme_precip, index=df.index)

        base_risk = pd.Series(0.0, index=df.index)
        for h, sev in hazards.items():
            base_risk += sev.fillna(0)
        base_risk = base_risk / max(len(hazards), 1)

        interaction_boost = pd.Series(0.0, index=df.index)
        detected_pairs = []
        for (h1, h2), config in self.interaction_matrix.items():
            s1 = hazards.get(h1, pd.Series(0, index=df.index))
            s2 = hazards.get(h2, pd.Series(0, index=df.index))
            both_active = (s1 > 50) & (s2 > 50)
            boost = both_active * (s1 + s2) / 200 * config["weight"] * 50
            interaction_boost += boost
            if both_active.any():
                detected_pairs.append({
                    "pair": f"{h1}+{h2}",
                    "name": config["name"],
                    "districts_affected": int(both_active.sum()),
                    "max_boost": float(boost.max()),
                })

        compound_severity = (base_risk + interaction_boost).clip(0, 100)
        out["compound_severity"] = compound_severity.round(1)
        out["compound_event"] = (compound_severity >= 50).astype(int)

        hazard_counts = pd.DataFrame(hazards).clip(0, 100)
        out["compound_hazard_count"] = (hazard_counts > 50).sum(axis=1)
        out["compound_class"] = self.classifier.classify(out["compound_severity"])
        out["compound_drivers"] = ""

        for idx in out.index:
            active = [h for h, s in hazards.items() if s.get(idx, 0) > 50]
            if active:
                out.at[idx, "compound_drivers"] = "+".join(active)

        self._detected_pairs = detected_pairs
        return out

    @property
    def active_interactions(self) -> List[Dict]:
        return getattr(self, "_detected_pairs", [])
