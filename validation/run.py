"""Orchestrates the full validation pipeline across models and districts."""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from config import CFG
from data.source_manager import DataSourceManager
from hazards.detection import HazardDetector
from validation.historical_validation import HistoricalValidator
from validation.skill_scores import SkillScoreCalculator as SkillScoreComputer
from validation.reliability import ReliabilityAnalyzer
from validation.confusion import ConfusionMatrixBuilder
from models.classical import ClassicalModels
from models.evaluator import ModelEvaluator
from utils import get_logger

logger = get_logger(__name__)


def run_district_validation(
    district: str,
    target: str = "tmax",
    source_manager: Optional[DataSourceManager] = None,
    detector: Optional[HazardDetector] = None,
) -> Dict:
    """Run full validation for a single district."""
    if source_manager is None:
        source_manager = DataSourceManager()
    if detector is None:
        detector = HazardDetector()

    logger.info(f"Validating {district} / {target}")

    data = source_manager.get_district_timeseries(district, ["tmax", "tmin", "precip"])
    if data is None or data.empty:
        return {"district": district, "error": "No data available"}

    lcc = detector.compute_climate_indices(data)
    hazards = detector.detect_all(lcc)

    validator = HistoricalValidator()
    validation_result = validator.validate_district(district, hazards)

    skill = SkillScoreComputer()
    skill_scores = {}
    if validation_result and "observations" in validation_result and "predictions" in validation_result:
        obs = validation_result["observations"]
        pred = validation_result["predictions"]
        skill_scores["brier"] = float(skill.brier_score(obs, pred))
        skill_scores["bss"] = float(skill.brier_skill_score(obs, pred))
        skill_scores["crps"] = float(skill.crps(obs, pred))

    reliability = ReliabilityAnalyzer()
    reliability_result = reliability.analyze(validation_result)

    confusion = ConfusionMatrixBuilder()
    conf_matrix = confusion.build(validation_result)
    conf_metrics = confusion.compute_metrics(conf_matrix)

    return {
        "district": district,
        "target": target,
        "validation": validation_result,
        "skill_scores": {k: float(v) for k, v in skill_scores.items() if isinstance(v, (int, float))},
        "reliability": reliability_result,
        "confusion": conf_metrics,
        "data_points": len(data),
        "hazard_events": int(hazards[["flood_event", "drought_event", "heatwave_event"]].sum().sum()),
    }


def run_all_validations(districts: Optional[List[str]] = None) -> Dict:
    """Run validation across all specified districts."""
    from geospatial.boundaries import DistrictBoundaries

    boundaries = DistrictBoundaries()
    districts = districts or boundaries.district_names[:5]

    source_manager = DataSourceManager()
    detector = HazardDetector()

    logger.info(f"Running validation for {len(districts)} districts")
    results = {}

    for district in districts:
        result = run_district_validation(district, source_manager=source_manager, detector=detector)
        results[district] = result
        logger.info(f"  {district}: {result.get('data_points', 0)} pts, {result.get('hazard_events', 0)} events")

    return results
