import numpy as np
import pandas as pd

from config import CFG
from hazards.categories import HazardClassifier
from utils.temporal_utils import consecutive_run

# IMD geography classification for MP districts
HILLY_DISTRICTS = set()
COASTAL_DISTRICTS = set()


def get_thresholds(district: str | None = None):
    """Return (abs_thresh, dep_thresh, severe_dep, abs_severe, label) based on geography."""
    base_dep = CFG.hazard.heatwave_departure  # 4.5
    severe_dep = CFG.hazard.heatwave_severe_departure  # 6.5
    if district and district.title() in COASTAL_DISTRICTS:
        return 37.0, base_dep, severe_dep, 42.0, "coastal"
    if district and district.title() in HILLY_DISTRICTS:
        return 30.0, 3.0, 5.0, 35.0, "hilly"
    return CFG.hazard.heatwave_tmax_threshold, base_dep, severe_dep, 47.0, "plains"


class HeatwaveDetector:
    def __init__(self):
        self.classifier = HazardClassifier()

    def detect(self, df: pd.DataFrame, district: str | None = None) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        if "tmax" not in df.columns:
            out["heatwave_severity"] = 0
            out["heatwave_event"] = 0
            out["heatwave_class"] = "Normal"
            out["heatwave_category"] = "No Heatwave"
            return out

        tmax = df["tmax"].fillna(0)
        clim_end = pd.Timestamp(CFG.climatology_end)
        abs_thresh, dep_thresh, severe_dep, abs_severe_thresh, geo = get_thresholds(district)

        train_tmax = tmax[tmax.index <= clim_end]
        doy_normal = train_tmax.groupby(train_tmax.index.dayofyear).mean()
        tmax_normal = tmax.index.dayofyear.map(doy_normal)
        departure = tmax - tmax_normal

        # IMD criteria per geography
        meets_abs = tmax >= abs_thresh
        meets_abs_severe = tmax >= abs_severe_thresh
        meets_dep = departure >= dep_thresh
        meets_severe_dep = departure >= severe_dep

        # Heatwave: meets absolute threshold AND departure threshold
        is_heatwave = meets_abs & meets_dep & ~meets_severe_dep
        is_heatwave = is_heatwave | meets_abs_severe

        # Severe: meets absolute threshold AND severe departure
        is_severe = meets_abs & meets_severe_dep
        is_severe = is_severe | (tmax >= abs_severe_thresh + 2)

        # Sustained 2+ consecutive days
        hw_sustained = consecutive_run(is_heatwave, min_run=2)
        sev_sustained = consecutive_run(is_severe, min_run=2)

        # Severity Score (0-100)
        severity = pd.Series(0.0, index=df.index)
        severity += ((tmax - abs_thresh + 5).clip(0, 12) / 12 * 30).clip(0, 30)
        severity += (departure.clip(0, 8) / 8 * 45).clip(0, 45)
        severity += (hw_sustained.astype(float) * 15)
        severity += (sev_sustained.astype(float) * 10)
        severity = np.where(tmax < abs_thresh - 5, 0, severity)

        # IMD Classification
        hw_category = pd.Series("No Heatwave", index=df.index)
        hw_category = np.where(sev_sustained, "Severe Heatwave",
                       np.where(hw_sustained, "Heatwave",
                       np.where(meets_abs, "Hot Day",
                       np.where(meets_dep, "Elevated Temperature", "Normal"))))

        out["heatwave_severity"] = pd.Series(severity, index=df.index).clip(0, 100).round(1)
        out["heatwave_event"] = (hw_sustained | sev_sustained).astype(int)
        out["heatwave_category"] = hw_category
        out["heatwave_departure"] = departure.round(2)
        out["heatwave_tmax_normal"] = tmax_normal.round(2)
        out["heatwave_class"] = np.where(sev_sustained, "Severe Heatwave",
                                  np.where(hw_sustained, "Heatwave",
                                  np.where(meets_abs, "Hot", "Normal")))
        return out
