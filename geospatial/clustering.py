from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.preprocessing import StandardScaler

from utils.logger import log


class HazardClustering:
    def __init__(self):
        self._clusters: Dict[str, List] = {}

    def dbscan_clustering(
        self,
        points: np.ndarray,
        severity_values: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> Dict:
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_scaled, sample_weight=severity_values)

        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        clusters = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            mask = clustering.labels_ == label
            clusters.append({
                "cluster_id": int(label),
                "n_points": int(mask.sum()),
                "mean_severity": float(np.mean(severity_values[mask])),
                "max_severity": float(np.max(severity_values[mask])),
                "centroid": list(np.mean(points[mask], axis=0)),
            })

        result = {
            "algorithm": "DBSCAN",
            "n_clusters": n_clusters,
            "n_noise": int((clustering.labels_ == -1).sum()),
            "clusters": sorted(clusters, key=lambda c: c["mean_severity"], reverse=True),
        }
        self._clusters["dbscan"] = result
        return result

    def kmeans_clustering(
        self,
        points: np.ndarray,
        n_clusters: int = 5,
        severity_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(points_scaled)

        clusters = []
        for label in range(n_clusters):
            mask = kmeans.labels_ == label
            severity = severity_weights[mask] if severity_weights is not None else np.ones(mask.sum())
            clusters.append({
                "cluster_id": label,
                "n_points": int(mask.sum()),
                "mean_severity": float(np.mean(severity)),
                "centroid": list(np.mean(points[mask], axis=0)),
            })

        result = {
            "algorithm": "KMeans",
            "n_clusters": n_clusters,
            "inertia": float(kmeans.inertia_),
            "clusters": sorted(clusters, key=lambda c: c["mean_severity"], reverse=True),
        }
        self._clusters["kmeans"] = result
        return result

    def get_hazard_zones(
        self,
        district_gdf,
        severity_by_district: Dict[str, float],
        n_zones: int = 4,
    ) -> pd.DataFrame:
        zones = pd.DataFrame([
            {"district": d, "severity": s}
            for d, s in severity_by_district.items()
        ])
        zones["zone"] = pd.qcut(zones["severity"], q=n_zones, labels=[f"Zone {i+1}" for i in range(n_zones)])
        return zones
