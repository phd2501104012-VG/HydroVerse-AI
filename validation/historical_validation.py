from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

from utils.logger import log


class HistoricalValidator:
    def __init__(self):
        self._events_db: Dict[str, pd.DataFrame] = {}

    def load_known_events(self, path: Optional[str] = None) -> pd.DataFrame:
        if path and pd.io.common.file_exists(path):
            df = pd.read_csv(path)
            log.info(f"Loaded {len(df)} known events from {path}")
            return df

        events = [
            {"district": "Bhopal", "hazard": "flood", "start": "2016-07-01", "end": "2016-07-15", "severity": "Severe"},
            {"district": "Indore", "hazard": "flood", "start": "2016-07-01", "end": "2016-07-10", "severity": "Severe"},
            {"district": "Jabalpur", "hazard": "flood", "start": "2019-08-01", "end": "2019-08-10", "severity": "Extreme"},
            {"district": "Gwalior", "hazard": "heatwave", "start": "2023-05-15", "end": "2023-05-25", "severity": "Severe"},
            {"district": "Bhopal", "hazard": "drought", "start": "2018-06-01", "end": "2018-09-30", "severity": "Severe"},
            {"district": "Rewa", "hazard": "drought", "start": "2017-05-01", "end": "2017-08-31", "severity": "Severe"},
            {"district": "Indore", "hazard": "heatwave", "start": "2024-05-20", "end": "2024-05-30", "severity": "Severe"},
        ]
        return pd.DataFrame(events)

    def validate(
        self,
        detected_episodes: pd.DataFrame,
        known_events: pd.DataFrame,
        hit_window_days: int = 5,
    ) -> Dict:
        hits = 0
        misses = 0
        false_alarms = 0
        hit_details = []
        miss_details = []
        false_alarm_details = []

        for _, event in known_events.iterrows():
            event_start = pd.Timestamp(event["start"])
            event_end = pd.Timestamp(event["end"])
            district = event["district"]
            hazard = event["hazard"]

            matched = detected_episodes[
                (detected_episodes["district"] == district)
                & (detected_episodes["hazard"] == hazard)
                & (detected_episodes["start"] <= event_end + pd.Timedelta(days=hit_window_days))
                & (detected_episodes["end"] >= event_start - pd.Timedelta(days=hit_window_days))
            ]

            if not matched.empty:
                hits += 1
                hit_details.append({
                    "event": f"{district}/{hazard} ({event['start']})",
                    "detected_start": str(matched.iloc[0]["start"]),
                    "peak_severity": float(matched.iloc[0]["peak_severity"]),
                })
            else:
                misses += 1
                miss_details.append({
                    "event": f"{district}/{hazard} ({event['start']})",
                    "known_severity": event["severity"],
                })

        for _, detected in detected_episodes.iterrows():
            d = detected["district"]
            h = detected["hazard"]
            ds = detected["start"]
            de = detected["end"]

            matched_known = known_events[
                (known_events["district"] == d)
                & (known_events["hazard"] == h)
                & (pd.to_datetime(known_events["start"]) <= de + pd.Timedelta(days=hit_window_days))
                & (pd.to_datetime(known_events["end"]) >= ds - pd.Timedelta(days=hit_window_days))
            ]

            if matched_known.empty:
                false_alarms += 1
                false_alarm_details.append({
                    "detected": f"{d}/{h} ({ds})",
                    "peak_severity": float(detected["peak_severity"]),
                })

        total_known = len(known_events)
        total_detected = len(detected_episodes)

        pod = hits / total_known if total_known > 0 else 0
        far = false_alarms / total_detected if total_detected > 0 else 0
        hss = 2 * (hits * (total_detected - false_alarms) - misses * false_alarms) / (
            (hits + misses) * (false_alarms + (total_detected - false_alarms))
            + (hits + false_alarms) * (misses + (total_detected - false_alarms))
        ) if total_known > 0 else 0

        return {
            "total_known_events": total_known,
            "total_detected": total_detected,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "probability_of_detection": round(pod, 3),
            "false_alarm_ratio": round(far, 3),
            "heinsske_skill_score": round(hss, 3),
            "hit_details": hit_details,
            "miss_details": miss_details,
            "false_alarm_details": false_alarm_details,
        }
