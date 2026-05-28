import numpy as np
import pandas as pd

from config import CFG
from hazards.categories import HazardClassifier


class FloodDetector:
    def __init__(self):
        self.classifier = HazardClassifier()

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        if "precip" not in df.columns:
            for c in ["flood_severity","flood_event","flood_class","flood_category","flood_risk"]:
                out[c] = 0.0 if c in ("flood_severity","flood_event") else "Normal"
            return out

        p = df["precip"].fillna(0).clip(lower=0)
        t = CFG.hazard

        # IMD Heavy Rainfall Categories
        IMD_LIGHT = t.imd_precip_light        # 15.6
        IMD_MODERATE = t.imd_precip_moderate   # 15.6
        IMD_HEAVY = t.imd_precip_heavy         # 64.5
        IMD_VERY_HEAVY = t.imd_precip_very_heavy  # 115.6
        IMD_EXTREME = t.imd_precip_extreme     # 204.5

        # ── Persistence Score (consecutive heavy rain days) ──
        is_heavy = (p >= IMD_HEAVY).astype(int)
        heavy_groups = (is_heavy != is_heavy.shift()).cumsum()
        heavy_run = is_heavy.groupby(heavy_groups).cumsum()

        persistence_score = pd.Series(0.0, index=df.index)
        persistence_score = np.where(
            heavy_run >= 3, 70,
            np.where(heavy_run == 2, 50, np.where(heavy_run == 1, 25, 0)),
        )
        persistence_score = pd.Series(persistence_score, index=df.index)

        # ── 3-Day Cumulative Score ──
        p_3d = p.rolling(3, min_periods=1).sum()
        cum_3d_score = pd.Series(0.0, index=df.index)
        cum_3d_score = np.where(
            p_3d >= 250, 80,
            np.where(p_3d >= 200, 70,
                     np.where(p_3d >= 150, 55,
                              np.where(p_3d >= IMD_HEAVY, 25, 0))),
        )
        cum_3d_score = pd.Series(cum_3d_score, index=df.index)

        # ── Antecedent Soil Saturation (7-day prior) ──
        p_7d_prior = p.shift(1).rolling(7, min_periods=1).sum()
        saturation_mult = (p_7d_prior / 100).clip(0, 1.2)

        # ── Consecutive Wet Days (CWD) for flood risk ──
        is_wet = (p >= 1.0).astype(int)
        wet_groups = (is_wet != is_wet.shift()).cumsum()
        cwd = is_wet.groupby(wet_groups).cumsum()

        # ── Base flood severity score ──
        base = np.maximum(persistence_score, cum_3d_score)
        saturation_boost = (base > 0) * (saturation_mult * 30)
        severity = (base + saturation_boost).clip(0, 100)

        # ── IMD Flood Risk Classification ──
        flood_risk = pd.Series("Low", index=df.index)
        flood_risk = np.where(p >= IMD_EXTREME, "Severe",
                     np.where(p >= IMD_VERY_HEAVY, "High",
                     np.where(p >= IMD_HEAVY, "Moderate", "Low")))
        # Upgrade to Severe if sustained heavy rain AND saturated soil
        severe_mask = (heavy_run >= 2) & (saturation_mult > 0.8) & (p >= IMD_HEAVY)
        flood_risk = pd.Series(np.where(severe_mask, "Severe", flood_risk), index=df.index)

        # Zero out pre-monsoon (Oct-May) unless actual extreme rain occurs
        is_premonsoon = df.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5])
        severity = np.where(is_premonsoon & (p < IMD_HEAVY * 0.5), 0, severity)

        out["flood_severity"] = pd.Series(severity, index=df.index).round(1)
        out["flood_risk"] = flood_risk
        out["flood_event"] = (flood_risk != "Low").astype(int)
        out["flood_cwd"] = cwd
        out["flood_3d_cumulative"] = p_3d.round(1)
        out["flood_class"] = self.classifier.classify(out["flood_severity"])
        return out
